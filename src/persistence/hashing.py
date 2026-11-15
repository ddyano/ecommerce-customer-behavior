"""Hashing utilities for the persistence layer (Phase 1, Step 1).

Provides:
  * hash_file                 chunked SHA-256 of a file
  * canonical_json_bytes      deterministic JSON bytes (also used for digests)
  * hash_cleaning_config      digest of the cleaning configuration
  * fingerprint_code          digest of the source files that determine the output
  * fingerprint_dataframe     type-aware LOGICAL fingerprint of a DataFrame

Local-only: reads the files it is given and nothing else. No network calls.
Error messages never include absolute paths or cell values.

Logical fingerprint rules (FINGERPRINT_VERSION 1)
-------------------------------------------------
* Values are hashed with type tags, never as display strings, so int 5, str "5",
  float 5.0 and bool True are all different.
* Object columns keep NaN, None and pd.NA distinct. pandas string dtypes and
  categoricals have ONE missing value, so all their missing forms hash as NaN.
* Floats hash via float.hex() (so -0.0 differs from 0.0); NaN is canonical.
* Numeric, boolean and naive datetime columns hash raw little-endian bytes and
  the dtype name is part of the column's identity (int32 differs from int64,
  datetime64[us] differs from datetime64[ns]).
* A categorical column equals a plain-text column holding the same values,
  regardless of category order or unused categories. An object column and a
  pandas string-dtype column holding the same text and missing values are equal.
* Column names, column order, row order and row count are included.
  The index is NOT hashed (the cache stores row_id as a column).
* Supported dtypes: bool, signed/unsigned integers, floats, naive datetime64
  (any unit), object, pandas StringDtype (both "string" and the default "str"),
  and categoricals. Anything else (nullable Int64/boolean, timedelta, tz-aware
  datetimes, ArrowDtype, ...) raises FingerprintError.
* Python's hash() is never used; the result is identical across runs and
  independent of the chunk size.

Privacy
-------
The OUTER digest covers every column (including identifier columns) and is the
only fingerprint value meant to be stored in a manifest. Per-column digests
exist for in-process diagnostics only (see differing_columns); they are hidden
from repr() and are NOT part of to_manifest_dict(). Never persist or print them.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

FINGERPRINT_VERSION: int = 1
CONFIG_HASH_VERSION: int = 1
CODE_FINGERPRINT_VERSION: int = 1
DEFAULT_CHUNK_ROWS: int = 100_000
DEFAULT_FILE_CHUNK_BYTES: int = 1024 * 1024

# The first-party modules that determine the annotated DataFrame: the import
# closure of the loader and the cleaning step (checked mechanically by a test).
DEFAULT_CODE_FILES: tuple[str, ...] = (
    "src/config.py",
    "src/ingestion/loader.py",
    "src/preprocessing/cleaning.py",
    "src/preprocessing/rules.py",
    "src/validation/overlap.py",
    "src/validation/quality.py",
)

# Cleaning-config fields whose order does not affect the result. Any other
# list-like field keeps its order (over-invalidating is the safe direction).
SET_LIKE_CLEANING_FIELDS: frozenset[str] = frozenset(
    {
        "non_product_stock_codes",
        "unidentifiable_stock_codes",
        "voucher_stock_code_prefixes",
    }
)

# Value tags for generic (object / string / categorical) columns.
_TAG_NAN = b"\x00"
_TAG_NONE = b"\x01"
_TAG_NA = b"\x02"
_TAG_STR = b"\x03"
_TAG_INT = b"\x04"
_TAG_FLOAT = b"\x05"
_TAG_BOOL = b"\x06"
_TAG_BYTES = b"\x07"
_BOOL_TRUE = _TAG_BOOL + b"\x01"
_BOOL_FALSE = _TAG_BOOL + b"\x00"

_COLUMN_PREFIX = b"column-v1"
_DATAFRAME_PREFIX = b"dataframe-v1"
_KIND_VALUES = "values"
_TYPED_FAMILIES = ("bool", "int", "float", "datetime")

_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")


class HashingError(Exception):
    """Base class for hashing errors."""


class FingerprintError(HashingError):
    """Raised when a DataFrame cannot be fingerprinted safely."""


class ConfigHashError(HashingError):
    """Raised when a configuration cannot be hashed safely."""


class _UnsupportedValue(Exception):
    """Internal: a cell value of a type the fingerprint does not support."""


def _lp(data: bytes) -> bytes:
    """Length-prefix a byte string."""
    return _U64.pack(len(data)) + data


def _check_positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}.")


# --------------------------------------------------------------------------
# Basic digests
# --------------------------------------------------------------------------
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, compact separators, ASCII, no NaN."""
    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HashingError(f"Value cannot be written as canonical JSON: {exc}") from exc
    return text.encode("ascii")


@dataclass(frozen=True)
class FileDigest:
    name: str
    size_bytes: int
    sha256: str


def hash_file(
    path: Path | str, chunk_bytes: int = DEFAULT_FILE_CHUNK_BYTES
) -> FileDigest:
    """SHA-256 of a file, read in chunks. The size is the number of bytes read."""
    _check_positive_int(chunk_bytes, "chunk_bytes")
    file_path = Path(path)
    if not file_path.is_file():
        raise HashingError(f"File not found or not a regular file: {file_path.name}")
    digest = hashlib.sha256()
    size = 0
    try:
        with file_path.open("rb") as handle:
            while True:
                block = handle.read(chunk_bytes)
                if not block:
                    break
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise HashingError(
            f"Could not read {file_path.name}: {type(exc).__name__}"
        ) from exc
    return FileDigest(file_path.name, size, digest.hexdigest())


# --------------------------------------------------------------------------
# Cleaning-configuration hash
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigHash:
    digest: str
    canonical: dict[str, Any]
    version: int


def _config_scalar(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigHashError(f"'{label}' is a non-finite number.")
        return value
    raise ConfigHashError(
        f"'{label}' has an unsupported type for hashing: {type(value).__name__}."
    )


def _config_value(value: Any, label: str, *, set_like: bool) -> Any:
    if isinstance(value, (tuple, list)):
        items = [_config_scalar(item, label) for item in value]
        if set_like:
            if not all(isinstance(item, str) for item in items):
                raise ConfigHashError(f"'{label}' must contain only strings.")
            items = sorted(items)
        return items
    return _config_scalar(value, label)


def _string_items(value: Any, label: str) -> list[str]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ConfigHashError(f"'{label}' must be a list of strings.")
    return list(value)


def hash_cleaning_config(
    cleaning: Any,
    required_columns: Sequence[str],
    sheet_names: Sequence[str] | None,
) -> ConfigHash:
    """Digest of the settings that influence the cleaned output.

    The fields are taken from the dataclass itself, so a field added later cannot
    be silently left out. Paths and comments are not part of these inputs.

    Ordering: the three stock-code lists are membership sets (they are only used
    with isin() or an OR over prefixes), so they are sorted before hashing.
    required_columns and sheet_names KEEP their order: their order is not proven
    irrelevant to the loaded or cleaned frame, so reordering them invalidates
    the cache (the safe direction).
    """
    if not dataclasses.is_dataclass(cleaning) or isinstance(cleaning, type):
        raise ConfigHashError("The cleaning configuration must be a dataclass instance.")
    fields: dict[str, Any] = {}
    for item in dataclasses.fields(cleaning):
        fields[item.name] = _config_value(
            getattr(cleaning, item.name),
            f"cleaning.{item.name}",
            set_like=item.name in SET_LIKE_CLEANING_FIELDS,
        )
    required = _string_items(required_columns, "schema.required_columns")
    sheets = (
        None
        if sheet_names is None
        else _string_items(sheet_names, "data.sheet_names")
    )
    canonical = {
        "config_hash_version": CONFIG_HASH_VERSION,
        "cleaning": fields,
        "required_columns": required,
        "sheet_names": sheets,
    }
    return ConfigHash(
        digest=sha256_hex(canonical_json_bytes(canonical)),
        canonical=canonical,
        version=CONFIG_HASH_VERSION,
    )


# --------------------------------------------------------------------------
# Code fingerprint
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CodeFingerprint:
    digest: str
    files: tuple[tuple[str, str], ...]  # (relative posix path, SHA-256 of normalised bytes)
    version: int


def fingerprint_code(
    project_root: Path | str,
    relative_paths: Sequence[str] = DEFAULT_CODE_FILES,
) -> CodeFingerprint:
    """Digest of source files. CRLF is normalised to LF so checkouts agree."""
    root = Path(project_root).resolve()
    requested = list(relative_paths)
    if not requested:
        raise HashingError("No source files were given to fingerprint.")
    entries: dict[str, str] = {}
    for item in requested:
        if not isinstance(item, str) or not item.strip():
            raise HashingError("Source file paths must be non-empty strings.")
        candidate = (root / item.replace("\\", "/")).resolve()
        try:
            recorded = candidate.relative_to(root).as_posix()
        except ValueError:
            raise HashingError(
                f"Source file is outside the project folder: {item!r}"
            ) from None
        if recorded in entries:
            raise HashingError(f"Duplicate source file: {recorded}")
        if not candidate.is_file():
            raise HashingError(f"Source file not found: {recorded}")
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            raise HashingError(
                f"Could not read {recorded}: {type(exc).__name__}"
            ) from exc
        entries[recorded] = sha256_hex(data.replace(b"\r\n", b"\n"))
    files = tuple(sorted(entries.items()))
    digest = sha256_hex(
        canonical_json_bytes(
            {
                "code_fingerprint_version": CODE_FINGERPRINT_VERSION,
                "files": [list(entry) for entry in files],
            }
        )
    )
    return CodeFingerprint(digest, files, CODE_FINGERPRINT_VERSION)


# --------------------------------------------------------------------------
# Logical DataFrame fingerprint
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DataFrameFingerprint:
    version: int
    digest: str  # OUTER digest over every column; the only value meant for a manifest
    rows: int
    columns: tuple[tuple[str, str], ...]  # (name, kind)
    # DIAGNOSTIC ONLY (see differing_columns). One-way digests of single columns,
    # including identifier columns: never persist or print them. Hidden from repr().
    column_digests: tuple[tuple[str, str], ...] = field(repr=False)

    def to_manifest_dict(self) -> dict[str, Any]:
        """The manifest-safe form: version, outer digest, row count, column kinds.

        Deliberately omits the per-column digests.
        """
        return {
            "fingerprint_version": self.version,
            "digest": self.digest,
            "rows": self.rows,
            "columns": [[name, kind] for name, kind in self.columns],
        }


def _encode_value(value: Any) -> bytes:
    """Strict, type-tagged encoding of one cell. Exact types only."""
    kind = type(value)
    if kind is str:
        data = value.encode("utf-8", "surrogatepass")
        return _TAG_STR + _U64.pack(len(data)) + data
    if kind is int:
        data = str(value).encode("ascii")
        return _TAG_INT + _U64.pack(len(data)) + data
    if kind is float:
        if value != value:
            return _TAG_NAN
        data = value.hex().encode("ascii")
        return _TAG_FLOAT + _U64.pack(len(data)) + data
    if kind is bool:
        return _BOOL_TRUE if value else _BOOL_FALSE
    if value is None:
        return _TAG_NONE
    if value is pd.NA:
        return _TAG_NA
    if kind is bytes:
        return _TAG_BYTES + _U64.pack(len(value)) + value
    raise _UnsupportedValue(kind.__name__)


def _encode_value_dtype_missing(value: Any) -> bytes:
    """For dtypes with ONE missing value (pandas string dtypes): every missing form is NaN.

    A NaN float of any float subclass (including numpy.float64) counts as missing here,
    because a string dtype cannot hold a float value.
    """
    if value is None or value is pd.NA or (isinstance(value, float) and value != value):
        return _TAG_NAN
    return _encode_value(value)


def _column_kind(name: str, dtype: Any) -> tuple[str, str]:
    """Return (kind recorded in the fingerprint, processing family).

    Only the dtypes listed in the module docstring are supported. pandas'
    is_string_dtype() is deliberately NOT used: it is also True for object and
    ArrowDtype(string), which must not be silently treated as text.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        return _KIND_VALUES, "categorical"
    if isinstance(dtype, pd.StringDtype):
        return _KIND_VALUES, "string"
    if isinstance(dtype, np.dtype):
        if dtype.kind == "O":
            return _KIND_VALUES, "object"
        if dtype.kind == "b":
            return "bool", "bool"
        if dtype.kind in "iu":
            return dtype.name, "int"
        if dtype.kind == "f":
            return dtype.name, "float"
        if dtype.kind == "M":
            return dtype.name, "datetime"
    raise FingerprintError(f"column {name!r}: unsupported dtype {dtype}")


def _typed_chunk_bytes(chunk: pd.Series, family: str) -> bytes:
    """Raw little-endian bytes of a typed chunk. Never writes into the input.

    pandas 3 returns READ-ONLY arrays from to_numpy(); every operation below
    either creates a new array or only reads.
    """
    values = chunk.to_numpy()
    if family == "bool":
        return values.astype(np.uint8).tobytes()
    if family == "datetime":
        # Every NaT is the minimum int64, so missing timestamps are canonical.
        return np.ascontiguousarray(values.view("i8"), dtype="<i8").tobytes()
    if family == "float":
        values = np.where(np.isnan(values), np.array(np.nan, dtype=values.dtype), values)
    return np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<")).tobytes()


def _category_table(name: str, dtype: pd.CategoricalDtype) -> list[bytes]:
    """Encoded bytes per category; the LAST entry is the missing value (code -1)."""
    try:
        table = [_encode_value(category) for category in dtype.categories.tolist()]
    except _UnsupportedValue as exc:
        raise FingerprintError(
            f"column {name!r}: unsupported category value type {exc.args[0]}"
        ) from None
    table.append(_TAG_NAN)
    return table


def _values_chunk_bytes(
    chunk: pd.Series,
    family: str,
    name: str,
    start: int,
    category_table: list[bytes] | None,
) -> bytes:
    if family == "categorical":
        codes = chunk.cat.codes.tolist()
        return b"".join(map(category_table.__getitem__, codes))
    values = chunk.tolist()
    encode = _encode_value_dtype_missing if family == "string" else _encode_value
    try:
        return b"".join([encode(value) for value in values])
    except _UnsupportedValue:
        for offset, value in enumerate(values):
            try:
                encode(value)
            except _UnsupportedValue:
                raise FingerprintError(
                    f"column {name!r}: unsupported value type "
                    f"{type(value).__name__} at row position {start + offset}"
                ) from None
        raise


def _fingerprint_column(
    series: pd.Series, name: str, kind: str, family: str, chunk_rows: int
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_COLUMN_PREFIX)
    hasher.update(_lp(kind.encode("ascii")))
    table = _category_table(name, series.dtype) if family == "categorical" else None
    for start in range(0, len(series), chunk_rows):
        chunk = series.iloc[start : start + chunk_rows]
        if family in _TYPED_FAMILIES:
            hasher.update(_typed_chunk_bytes(chunk, family))
        else:
            hasher.update(_values_chunk_bytes(chunk, family, name, start, table))
    return hasher.digest()


def fingerprint_dataframe(
    df: pd.DataFrame,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    column_seconds: dict[str, float] | None = None,
) -> DataFrameFingerprint:
    """Type-aware logical fingerprint (see the module docstring for the rules).

    Memory is bounded by `chunk_rows`, not by the number of rows. The byte stream
    of every column is a plain concatenation of per-row records with no per-chunk
    headers, so the result does not depend on `chunk_rows`. If `column_seconds`
    is given it is filled with the seconds spent per column.
    """
    _check_positive_int(chunk_rows, "chunk_rows")
    if not isinstance(df, pd.DataFrame):
        raise FingerprintError("fingerprint_dataframe expects a pandas DataFrame.")
    names = list(df.columns)
    if not all(isinstance(name, str) for name in names):
        raise FingerprintError("DataFrame column names must be strings.")
    if len(set(names)) != len(names):
        raise FingerprintError("DataFrame contains duplicate column names.")

    outer = hashlib.sha256()
    outer.update(_DATAFRAME_PREFIX)
    outer.update(_U32.pack(FINGERPRINT_VERSION))
    outer.update(_U64.pack(len(df)))
    outer.update(_U64.pack(len(names)))

    kinds: list[tuple[str, str]] = []
    digests: list[tuple[str, str]] = []
    for position, name in enumerate(names):
        series = df.iloc[:, position]
        kind, family = _column_kind(name, series.dtype)
        started = time.perf_counter()
        column_digest = _fingerprint_column(series, name, kind, family, chunk_rows)
        if column_seconds is not None:
            column_seconds[name] = time.perf_counter() - started
        outer.update(_lp(name.encode("utf-8", "surrogatepass")))
        outer.update(_lp(kind.encode("ascii")))
        outer.update(column_digest)
        kinds.append((name, kind))
        digests.append((name, column_digest.hex()))
    return DataFrameFingerprint(
        version=FINGERPRINT_VERSION,
        digest=outer.hexdigest(),
        rows=int(len(df)),
        columns=tuple(kinds),
        column_digests=tuple(digests),
    )


def differing_columns(
    first: DataFrameFingerprint, second: DataFrameFingerprint
) -> tuple[str, ...]:
    """Column NAMES whose fingerprint differs, plus columns present in only one.

    Uses the per-column digests in memory only; returns names, never digests.
    """
    first_info = {
        name: (kind, digest)
        for (name, kind), (_, digest) in zip(first.columns, first.column_digests)
    }
    second_info = {
        name: (kind, digest)
        for (name, kind), (_, digest) in zip(second.columns, second.column_digests)
    }
    changed = [name for name, info in first_info.items() if second_info.get(name) != info]
    changed.extend(name for name in second_info if name not in first_info)
    return tuple(changed)