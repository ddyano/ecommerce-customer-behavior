"""Single-file Parquet codec for validated annotated transaction frames.

This is a codec only: it does not manage caches, register external
expectations, or write sidecar files.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.persistence.hashing import canonical_json_bytes, fingerprint_dataframe
from src.persistence.schema import (
    ANNOTATED_COLUMNS,
    CATEGORICAL_COLUMNS,
    DYNAMIC_CATEGORICALS,
    PersistenceView,
    RULE_CATEGORICALS,
    SCHEMA_VERSION,
    SchemaError,
    schema_digest,
    to_annotated_view,
    to_persistence_view,
    validate_persistence_view,
)
from src.persistence.text_exceptions import (
    MANIFEST_FIELDS,
    TextExceptionError,
    encode_descriptions,
    restore_descriptions,
    validate_description_manifest,
)

_METADATA_KEY = b"ecommerce.persistence.codec.v1"
_METADATA_KEYS = frozenset(
    {
        "codec_version",
        "schema_version",
        "schema_digest",
        "original_dtypes",
        "dynamic_categories",
        "dynamic_nulls",
        "description_dtype",
        "description_null_kinds",
        "exception_manifest",
        "fingerprint",
    }
)
_CODEC_VERSION = 1


class CodecError(ValueError):
    """A Parquet artifact cannot be safely encoded, decoded, or verified."""


@dataclass(frozen=True)
class VerificationResult:
    rows: int
    columns: int
    schema_version: int
    fingerprint: str


def _raise_safe(message: str) -> None:
    raise CodecError(message) from None


def _json_safe(value: Any) -> Any:
    """Convert schema metadata to canonical-JSON-compatible builtins."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            _raise_safe("Metadata contains an unsupported non-finite value.")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            _raise_safe("Metadata contains an invalid key.")
        return {key: _json_safe(item) for key, item in value.items()}
    _raise_safe("Metadata contains an unsupported value.")


def _manifest_records(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    if tuple(manifest.columns) != MANIFEST_FIELDS:
        _raise_safe("Exception manifest fields do not match the approved schema.")
    result: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False, name=None):
        row_id, original_type, text, source_sheet, stock_code = row
        record = {
            "row_id": int(row_id),
            "original_type": str(original_type),
            "text": str(text),
            "source_sheet": str(source_sheet),
            "stock_code": None if pd.isna(stock_code) else str(stock_code),
        }
        result.append(record)
    return result


def _metadata_payload(frame: pd.DataFrame, persistence: PersistenceView, encoded) -> dict[str, Any]:
    return {
        "codec_version": _CODEC_VERSION,
        "schema_version": persistence.schema_version,
        "schema_digest": schema_digest(),
        "original_dtypes": _json_safe(persistence.original_dtypes),
        "dynamic_categories": _json_safe(persistence.dynamic_categories),
        "dynamic_nulls": _json_safe(persistence.dynamic_nulls),
        "description_dtype": _json_safe(dict(persistence.original_dtypes)["Description"]),
        "description_null_kinds": _json_safe(encoded.null_kinds),
        "exception_manifest": _manifest_records(encoded.manifest),
        "fingerprint": fingerprint_dataframe(frame).to_manifest_dict(),
    }


def _read_metadata(raw: bytes | None) -> dict[str, Any]:
    if raw is None:
        _raise_safe("Parquet artifact is missing codec metadata.")

    def unique_object(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key")
            output[key] = value
        return output

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        if not isinstance(payload, dict) or set(payload) != _METADATA_KEYS:
            _raise_safe("Parquet codec metadata fields are malformed.")
        if canonical_json_bytes(payload) != raw:
            _raise_safe("Parquet codec metadata is not canonical.")
    except CodecError:
        raise
    except Exception:
        _raise_safe("Parquet codec metadata is malformed.")
    if type(payload["codec_version"]) is not int or payload["codec_version"] != _CODEC_VERSION:
        _raise_safe("Unsupported Parquet codec version.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        _raise_safe("Unsupported annotated schema version.")
    if payload["schema_digest"] != schema_digest():
        _raise_safe("Parquet schema digest does not match this implementation.")
    if not isinstance(payload["fingerprint"], dict):
        _raise_safe("Parquet fingerprint metadata is malformed.")
    fp = payload["fingerprint"]
    if (
        set(fp) != {"fingerprint_version", "digest", "rows", "columns"}
        or type(fp.get("digest")) is not str
        or len(fp["digest"]) != 64
        or type(fp.get("rows")) is not int
        or fp["rows"] < 0
    ):
        _raise_safe("Parquet fingerprint metadata is malformed.")
    if not isinstance(fp["columns"], list):
        _raise_safe("Parquet fingerprint metadata is malformed.")
    return payload


def _strict_dtype(descriptor: Any, column: str) -> Any:
    """Decode supported Description dtype descriptors without coercion."""
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("kind"), str):
        _raise_safe("Description restoration metadata is malformed.")
    kind = descriptor["kind"]
    try:
        if kind == "object" and set(descriptor) == {"kind"}:
            return np.dtype(object)
        if kind == "string" and set(descriptor) == {"kind", "name", "storage", "na_value"}:
            missing = {"pd.NA": pd.NA, "nan": np.nan}[descriptor["na_value"]]
            dtype = pd.StringDtype(storage=descriptor["storage"], na_value=missing)
            if dtype.name != descriptor["name"]:
                raise ValueError
            return dtype
        if kind == "categorical" and set(descriptor) == {"kind", "categories", "ordered"}:
            return pd.CategoricalDtype(
                categories=descriptor["categories"], ordered=descriptor["ordered"]
            )
        if kind in {"int", "float", "datetime", "bool"} and set(descriptor) == {"kind", "name"}:
            dtype = np.dtype(descriptor["name"])
            expected = {"i": "int", "u": "int", "f": "float", "M": "datetime", "b": "bool"}.get(dtype.kind)
            if expected != kind:
                raise ValueError
            return dtype
    except Exception:
        _raise_safe(f"Description dtype metadata for {column!r} is unsupported.")
    _raise_safe(f"Description dtype metadata for {column!r} is malformed.")


def _validate_payload_shapes(payload: dict[str, Any], rows: int) -> None:
    if not isinstance(payload["original_dtypes"], list) or len(payload["original_dtypes"]) != len(ANNOTATED_COLUMNS):
        _raise_safe("Restoration dtype metadata is malformed.")
    for expected, item in zip(ANNOTATED_COLUMNS, payload["original_dtypes"]):
        if not isinstance(item, list) or len(item) != 2 or item[0] != expected:
            _raise_safe("Restoration dtype metadata has inconsistent columns.")
    if not isinstance(payload["dynamic_categories"], list) or len(payload["dynamic_categories"]) != len(DYNAMIC_CATEGORICALS):
        _raise_safe("Dynamic category metadata is malformed.")
    for expected_name, item in zip(DYNAMIC_CATEGORICALS, payload["dynamic_categories"]):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or item[0] != expected_name
            or not isinstance(item[1], list)
            or not all(type(value) is str for value in item[1])
            or item[1] != sorted(set(item[1]))
        ):
            _raise_safe("Dynamic category metadata is malformed.")
    if not isinstance(payload["dynamic_nulls"], list) or len(payload["dynamic_nulls"]) != len(DYNAMIC_CATEGORICALS):
        _raise_safe("Dynamic null metadata is malformed.")
    if not isinstance(payload["description_null_kinds"], list):
        _raise_safe("Description null metadata is malformed.")
    for item in payload["description_null_kinds"]:
        if (
            not isinstance(item, list) or len(item) != 2
            or type(item[0]) is not int or item[1] not in {"none", "pd.NA", "nan", "NaT"}
        ):
            _raise_safe("Description null metadata is malformed.")
    manifest = payload["exception_manifest"]
    if not isinstance(manifest, list):
        _raise_safe("Exception manifest is malformed.")
    for record in manifest:
        if not isinstance(record, dict) or set(record) != set(MANIFEST_FIELDS):
            _raise_safe("Exception manifest fields are malformed.")
        if (
            type(record["row_id"]) is not int
            or any(type(record[key]) is not str for key in ("original_type", "text", "source_sheet"))
            or (record["stock_code"] is not None and type(record["stock_code"]) is not str)
        ):
            _raise_safe("Exception manifest values are malformed.")
    if payload["fingerprint"]["rows"] != rows:
        _raise_safe("Parquet row count disagrees with fingerprint metadata.")
    if payload["description_dtype"] != dict(payload["original_dtypes"])["Description"]:
        _raise_safe("Description dtype metadata is inconsistent.")


def _manifest_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=MANIFEST_FIELDS)
    if frame.empty:
        return frame.astype(
            {
                "row_id": "int64",
                "original_type": "object",
                "text": "string",
                "source_sheet": "string",
                "stock_code": "object",
            }
        )
    frame["row_id"] = frame["row_id"].astype("int64")
    for name in ("original_type", "text", "source_sheet"):
        frame[name] = frame[name].astype("string")
    frame["stock_code"] = pd.Series(frame["stock_code"].tolist(), dtype=object)
    return frame


def _rebuild_persistence_view(frame: pd.DataFrame, payload: dict[str, Any]) -> PersistenceView:
    try:
        original_dtypes = tuple((item[0], item[1]) for item in payload["original_dtypes"])
        dynamic_categories = tuple(
            (item[0], tuple(item[1])) for item in payload["dynamic_categories"]
        )
        dynamic_nulls = tuple(
            (item[0], tuple((entry[0], entry[1]) for entry in item[1]))
            for item in payload["dynamic_nulls"]
        )
        # Rebuild categories exactly from the versioned rule vocabulary and
        # metadata. Parquet dictionary categories are checked before this step.
        categories = {
            **RULE_CATEGORICALS,
            **dict(dynamic_categories),
        }
        for name in CATEGORICAL_COLUMNS:
            series = frame[name]
            values = series.astype(object).tolist()
            if any(pd.notna(value) and type(value) is not str for value in values):
                _raise_safe("Parquet categorical values have an invalid type.")
            allowed = categories[name]
            if any(pd.notna(value) and value not in allowed for value in values):
                _raise_safe("Parquet categorical values disagree with metadata.")
            frame[name] = pd.Categorical(values, categories=list(allowed), ordered=False)
        result = PersistenceView(
            frame=frame.loc[:, list(ANNOTATED_COLUMNS)],
            schema_version=payload["schema_version"],
            original_dtypes=original_dtypes,
            dynamic_categories=dynamic_categories,
            dynamic_nulls=dynamic_nulls,
        )
        validate_persistence_view(result)
        return result
    except CodecError:
        raise
    except (SchemaError, Exception):
        _raise_safe("Parquet data does not match persistence schema metadata.")


def _decode_table(table: pa.Table, payload: dict[str, Any]) -> pd.DataFrame:
    names = tuple(table.column_names)
    if names != ANNOTATED_COLUMNS:
        _raise_safe("Parquet columns do not match the versioned schema order.")
    _validate_payload_shapes(payload, table.num_rows)
    for name in CATEGORICAL_COLUMNS:
        field = table.schema.field(name)
        expected = RULE_CATEGORICALS.get(name)
        if expected is None:
            expected = dict((key, tuple(values)) for key, values in payload["dynamic_categories"])[name]
        if pa.types.is_null(field.type):
            if not (
                table.num_rows == 0
                and name in DYNAMIC_CATEGORICALS
                and not expected
            ):
                _raise_safe("Parquet categorical column is not dictionary encoded.")
            continue
        if not pa.types.is_dictionary(field.type):
            _raise_safe("Parquet categorical column is not dictionary encoded.")
        if not (pa.types.is_string(field.type.value_type) or pa.types.is_large_string(field.type.value_type)):
            _raise_safe("Parquet categorical dictionary has an invalid value type.")
        for chunk in table[name].chunks:
            if not pa.types.is_dictionary(chunk.type):
                _raise_safe("Parquet categorical chunks have inconsistent types.")
            if tuple(chunk.dictionary.to_pylist()) != tuple(expected):
                _raise_safe("Parquet categorical dictionary disagrees with metadata.")
    try:
        frame = table.to_pandas()
    except Exception:
        _raise_safe("Parquet values could not be decoded safely.")
    # Description is deliberately stored as Arrow string; exact source dtype
    # and null sentinels are restored by the exception API and metadata.
    desc = frame["Description"]
    if not (pa.types.is_string(table.schema.field("Description").type) or pa.types.is_large_string(table.schema.field("Description").type)):
        _raise_safe("Parquet Description column is not stored as text.")
    if any(not pd.isna(value) and type(value) is not str for value in desc.tolist()):
        _raise_safe("Parquet Description contains a malformed encoded value.")
    frame["Description"] = pd.Series(
        pd.array(desc.tolist(), dtype=pd.StringDtype(storage="pyarrow"))
    )
    # Restore exact source dtypes for non-categorical, non-Description fields.
    descriptors = dict((item[0], item[1]) for item in payload["original_dtypes"])
    for name in ANNOTATED_COLUMNS:
        if name == "Description" or name in CATEGORICAL_COLUMNS:
            continue
        dtype = _strict_dtype(descriptors[name], name)
        try:
            converted = frame[name].astype(dtype)
            # Reject value-changing casts by round-tripping through Python
            # scalar forms; the final full-frame fingerprint is authoritative.
            frame[name] = converted
        except Exception:
            _raise_safe("Parquet column cannot be restored to its recorded dtype.")
    frame = frame.loc[:, list(ANNOTATED_COLUMNS)]
    return frame


def _decode_path(path: Path) -> tuple[pd.DataFrame, VerificationResult]:
    try:
        parquet = pq.ParquetFile(path)
        try:
            table = parquet.read()
        finally:
            parquet.close()
        raw = (table.schema.metadata or {}).get(_METADATA_KEY)
        payload = _read_metadata(raw)
        if table.num_rows != payload["fingerprint"]["rows"]:
            _raise_safe("Parquet row count disagrees with fingerprint metadata.")
        _validate_payload_shapes(payload, table.num_rows)
        stored = _decode_table(table, payload)
        manifest = _manifest_frame(payload["exception_manifest"])
        null_kinds = tuple((item[0], item[1]) for item in payload["description_null_kinds"])
        original_dtypes = tuple((item[0], item[1]) for item in payload["original_dtypes"])
        description_dtype = _strict_dtype(payload["description_dtype"], "Description")
        exception_storage = stored.loc[:, ["row_id", "Description", "source_sheet", "StockCode"]].copy()
        exception_storage["source_sheet"] = exception_storage["source_sheet"].astype(object)
        exception_storage["StockCode"] = exception_storage["StockCode"].astype(object)
        validate_description_manifest(exception_storage, manifest)
        restored_description = restore_descriptions(
            exception_storage,
            manifest,
            null_kinds=null_kinds,
            original_dtype=description_dtype,
        )
        persisted_frame = stored.copy()
        persisted_frame["Description"] = restored_description
        persistence = _rebuild_persistence_view(persisted_frame, payload)
        annotated = to_annotated_view(persistence)
        fingerprint = fingerprint_dataframe(annotated).to_manifest_dict()
        if fingerprint != payload["fingerprint"]:
            _raise_safe("Restored frame fingerprint does not match the Parquet artifact.")
        result = VerificationResult(
            rows=len(annotated),
            columns=len(annotated.columns),
            schema_version=SCHEMA_VERSION,
            fingerprint=fingerprint["digest"],
        )
        return annotated, result
    except CodecError:
        raise
    except (SchemaError, TextExceptionError):
        _raise_safe("Parquet artifact failed schema or restoration validation.")
    except Exception:
        _raise_safe("Parquet artifact is corrupt or unsupported.")


def encode_parquet(frame: pd.DataFrame, path: Path | str) -> VerificationResult:
    """Atomically encode a validated annotated DataFrame to one Parquet file.

    The temporary file is written and fully decoded/verified before replacing
    the destination. A failure leaves any existing destination untouched.
    """
    target = Path(path)
    if not target.parent.is_dir():
        _raise_safe("Parquet destination directory does not exist.")
    try:
        persistence = to_persistence_view(frame)
        exceptions = encode_descriptions(frame)
        if exceptions.storage["row_id"].tolist() != frame["row_id"].tolist():
            _raise_safe("Description exception rows do not match the annotated frame.")
        storage = persistence.frame.copy(deep=True)
        storage["Description"] = pd.Series(
            exceptions.storage["Description"].array, index=storage.index
        )
        storage = storage.loc[:, list(ANNOTATED_COLUMNS)]
        payload = _metadata_payload(frame, persistence, exceptions)
        metadata_bytes = canonical_json_bytes(payload)
        table = pa.Table.from_pandas(storage, preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), _METADATA_KEY: metadata_bytes})
        # Require Arrow dictionary fields for categorical storage.
        for name in CATEGORICAL_COLUMNS:
            if not pa.types.is_dictionary(table.schema.field(name).type):
                _raise_safe("Categorical data could not be encoded without loss.")

        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            pq.write_table(table, temporary, version="2.6", compression="zstd", store_schema=True)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            restored, result = _decode_path(temporary)
            expected = fingerprint_dataframe(frame).to_manifest_dict()
            if fingerprint_dataframe(restored).to_manifest_dict() != expected:
                _raise_safe("Parquet round trip would alter annotated data.")
            os.replace(temporary, target)
            return result
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    except CodecError:
        raise
    except (SchemaError, TextExceptionError):
        _raise_safe("Annotated frame cannot be encoded by the Parquet codec.")
    except Exception:
        _raise_safe("Parquet encoding failed; destination was not replaced.")


def decode_parquet(path: Path | str) -> pd.DataFrame:
    """Decode and fully validate an annotated frame from a Parquet artifact."""
    artifact = Path(path)
    if not artifact.is_file():
        _raise_safe("Parquet artifact is missing or is not a regular file.")
    return _decode_path(artifact)[0]


def verify_parquet(path: Path | str) -> VerificationResult:
    """Verify artifact structure, metadata, restoration, and logical digest."""
    artifact = Path(path)
    if not artifact.is_file():
        _raise_safe("Parquet artifact is missing or is not a regular file.")
    return _decode_path(artifact)[1]
