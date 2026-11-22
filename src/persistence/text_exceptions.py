"""Lossless in-memory encoding for non-string Description scalars.

The exception manifest has a deliberately narrow, exact field allowlist. This
module performs no file I/O and does not implement a storage backend.
"""
from __future__ import annotations

import base64
import binascii
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.persistence.schema import (
    SchemaError,
    validate_annotated_view,
)

MANIFEST_FIELDS: tuple[str, ...] = (
    "row_id",
    "original_type",
    "text",
    "source_sheet",
    "stock_code",
)
_ENCODED_COLUMNS: tuple[str, ...] = ("row_id", "Description", "source_sheet", "StockCode")
_TYPE_TAGS = frozenset({"int", "float", "bool", "bytes"})


class TextExceptionError(ValueError):
    """Description exception data or its row mapping is invalid.

    Messages intentionally contain no cell values or identifiers.
    """


@dataclass(frozen=True)
class EncodedDescriptions:
    """String-compatible Description data plus private restoration metadata."""

    storage: pd.DataFrame
    manifest: pd.DataFrame
    original_dtype: Any
    null_kinds: tuple[tuple[int, str], ...]


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return type(value) is float and math.isnan(value)


def _encode_scalar(value: Any) -> tuple[str, str]:
    if type(value) is bool:
        return "bool", "true" if value else "false"
    if type(value) is int:
        return "int", str(value)
    if type(value) is float:
        return "float", value.hex()
    if type(value) is bytes:
        return "bytes", base64.b64encode(value).decode("ascii")
    raise TextExceptionError("Description contains an unsupported exceptional scalar type.")


def _decode_scalar(type_tag: Any, text: Any) -> Any:
    if type(type_tag) is not str or type_tag not in _TYPE_TAGS or type(text) is not str:
        raise TextExceptionError("Manifest contains an unsupported type tag or text value.")
    try:
        if type_tag == "bool":
            if text not in {"true", "false"}:
                raise ValueError
            return text == "true"
        if type_tag == "int":
            value = int(text, 10)
            if str(value) != text:
                raise ValueError
            return value
        if type_tag == "float":
            value = float.fromhex(text)
            if value.hex() != text:
                raise ValueError
            return value
        raw = base64.b64decode(text.encode("ascii"), validate=True)
        if base64.b64encode(raw).decode("ascii") != text:
            raise ValueError
        return raw
    except (ValueError, OverflowError, UnicodeEncodeError, binascii.Error):
        raise TextExceptionError("Manifest contains a malformed scalar encoding.") from None


def _validate_annotated(frame: Any) -> pd.DataFrame:
    try:
        validate_annotated_view(frame)
    except SchemaError as exc:
        # Do not relay schema messages that may mention source values.
        raise TextExceptionError("Annotated frame does not satisfy the persistence schema.") from None
    ids = frame["row_id"]
    if ids.isna().any() or ids.duplicated().any() or bool((ids < 0).any()):
        raise TextExceptionError("Annotated frame has missing or duplicate row mappings.")
    return frame


def encode_descriptions(annotated: pd.DataFrame) -> EncodedDescriptions:
    """Return storage-compatible text data and its exact five-field manifest.

    The encoded table contains only row_id, Description, source_sheet, and
    StockCode. Other source columns, including Invoice and Customer ID, are
    never accessed or returned. Null markers and the Description dtype are
    carried separately from the manifest to preserve null identity in memory.
    """
    frame = _validate_annotated(annotated)
    row_ids = frame["row_id"].tolist()
    descriptions = frame["Description"].tolist()
    encoded_values: list[Any] = []
    null_kinds: list[tuple[int, str]] = []
    manifest_rows: list[dict[str, Any]] = []
    for position, (row_id, value) in enumerate(zip(row_ids, descriptions)):
        if _is_null(value):
            if value is None:
                marker = "none"
            elif value is pd.NA:
                marker = "pd.NA"
            elif value is pd.NaT:
                marker = "NaT"
            else:
                marker = "nan"
            null_kinds.append((int(row_id), marker))
            encoded_values.append(pd.NA)
        elif type(value) is str:
            encoded_values.append(value)
        else:
            type_tag, text = _encode_scalar(value)
            encoded_values.append(text)
            manifest_rows.append(
                {
                    "row_id": int(row_id),
                    "original_type": type_tag,
                    "text": text,
                    "source_sheet": frame["source_sheet"].iloc[position],
                    "stock_code": frame["StockCode"].iloc[position],
                }
            )

    storage = pd.DataFrame(
        {
            "row_id": pd.Series(row_ids, dtype="int64"),
            "Description": pd.Series(encoded_values, dtype=pd.StringDtype(storage="pyarrow")),
            "source_sheet": frame["source_sheet"].reset_index(drop=True).copy(),
            "StockCode": frame["StockCode"].reset_index(drop=True).copy(),
        }
    )
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_FIELDS)
    if manifest.empty:
        manifest = manifest.astype(
            {
                "row_id": "int64",
                "original_type": "string",
                "text": "string",
                "source_sheet": frame["source_sheet"].dtype,
                "stock_code": frame["StockCode"].dtype,
            }
        )
    return EncodedDescriptions(
        storage=storage,
        manifest=manifest,
        original_dtype=frame["Description"].dtype,
        null_kinds=tuple(null_kinds),
    )


def _check_storage(storage: Any) -> pd.DataFrame:
    if not isinstance(storage, pd.DataFrame):
        raise TextExceptionError("Encoded Description data must be a DataFrame.")
    if tuple(storage.columns) != _ENCODED_COLUMNS or len(set(storage.columns)) != len(storage.columns):
        raise TextExceptionError("Encoded Description columns are malformed.")
    if storage["row_id"].dtype != np.dtype("int64") or storage["row_id"].isna().any():
        raise TextExceptionError("Encoded Description row mappings are malformed.")
    if storage["row_id"].duplicated().any():
        raise TextExceptionError("Encoded Description contains duplicate row mappings.")
    if not isinstance(storage["Description"].dtype, pd.StringDtype):
        raise TextExceptionError("Encoded Description data must use a pandas string dtype.")
    if storage["source_sheet"].isna().any():
        raise TextExceptionError("Encoded Description row metadata contains null values.")
    if not storage["source_sheet"].map(lambda value: type(value) is str).all():
        raise TextExceptionError("Encoded Description source-sheet metadata has an invalid type.")
    if not storage["StockCode"].map(
        lambda value: _is_null(value) or type(value) is str
    ).all():
        raise TextExceptionError("Encoded Description stock-code metadata has an invalid type.")
    return storage


def _check_manifest(manifest: Any) -> pd.DataFrame:
    if not isinstance(manifest, pd.DataFrame):
        raise TextExceptionError("Exception manifest must be a DataFrame.")
    if tuple(manifest.columns) != MANIFEST_FIELDS or len(set(manifest.columns)) != len(manifest.columns):
        raise TextExceptionError("Exception manifest fields are malformed.")
    if manifest["row_id"].dtype != np.dtype("int64") or manifest["row_id"].isna().any():
        raise TextExceptionError("Exception manifest row mappings are malformed.")
    if manifest["row_id"].duplicated().any():
        raise TextExceptionError("Exception manifest contains duplicate row mappings.")
    for column in ("original_type", "text", "source_sheet"):
        if manifest[column].isna().any():
            raise TextExceptionError("Exception manifest contains missing required values.")
    if not manifest["original_type"].map(lambda value: type(value) is str and value in _TYPE_TAGS).all():
        raise TextExceptionError("Exception manifest contains unsupported original types.")
    for value in manifest["text"].array:
        if type(value) is not str:
            raise TextExceptionError("Exception manifest text values must be strings.")
    for value in manifest["source_sheet"].array:
        if type(value) is not str:
            raise TextExceptionError("Exception manifest source-sheet values must be strings.")
    for value in manifest["stock_code"].array:
        if value is not None and not _is_null(value) and type(value) is not str:
            raise TextExceptionError("Exception manifest stock-code values have an invalid type.")
    # Validate canonical reversible encodings without returning decoded values.
    for type_tag, text in zip(manifest["original_type"], manifest["text"]):
        _decode_scalar(type_tag, text)
    return manifest


def validate_description_manifest(
    storage: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    original: pd.DataFrame | None = None,
) -> bool:
    """Validate mappings and consistency; return whether completeness was verified.

    Completeness is checked only when the original annotated frame is supplied.
    Without it, `False` means the manifest is structurally consistent but its
    completeness is unknown, not that full integrity has been established.
    """
    encoded = _check_storage(storage)
    entries = _check_manifest(manifest)
    storage_by_id = encoded.set_index("row_id", drop=False)
    manifest_ids = set(entries["row_id"].tolist())
    storage_ids = set(storage_by_id.index.tolist())
    if not manifest_ids <= storage_ids:
        raise TextExceptionError("Exception manifest contains missing or unknown row mappings.")
    for row in entries.itertuples(index=False, name=None):
        row_id, type_tag, text, source_sheet, stock_code = row
        stored = storage_by_id.loc[row_id]
        stored_description = stored["Description"]
        if _is_null(stored_description) or type(stored_description) is not str:
            raise TextExceptionError("Manifest references a null or malformed encoded Description value.")
        if stored_description != text:
            raise TextExceptionError("Exception manifest text disagrees with encoded row data.")
        if stored["source_sheet"] != source_sheet or not _same_nullable(
            stored["StockCode"], stock_code
        ):
            raise TextExceptionError("Exception manifest row metadata disagrees with encoded data.")

    if original is None:
        return False
    source = _validate_annotated(original)
    source_by_id = source.set_index("row_id", drop=False)
    if storage_ids != set(source_by_id.index.tolist()):
        raise TextExceptionError("Original and encoded row mappings do not match.")
    expected: dict[int, tuple[str, str]] = {}
    for row_id, value in zip(source["row_id"].tolist(), source["Description"].tolist()):
        if not _is_null(value) and type(value) is not str:
            expected[int(row_id)] = _encode_scalar(value)
    actual = {int(row.row_id): (row.original_type, row.text) for row in entries.itertuples(index=False)}
    if actual != expected:
        raise TextExceptionError("Exception manifest does not completely match the original frame.")
    for row_id, row in source_by_id.iterrows():
        stored = storage_by_id.loc[row_id]
        if stored["source_sheet"] != row["source_sheet"] or not _same_nullable(
            stored["StockCode"], row["StockCode"]
        ):
            raise TextExceptionError("Original and encoded row metadata do not match.")
        expected_text = row["Description"] if type(row["Description"]) is str else (
            _encode_scalar(row["Description"])[1]
            if not _is_null(row["Description"])
            else pd.NA
        )
        stored_text = stored["Description"]
        if _is_null(expected_text):
            if not _is_null(stored_text):
                raise TextExceptionError("Original and encoded Description nulls do not match.")
        elif stored_text != expected_text:
            raise TextExceptionError("Original and encoded Description values do not match.")
    return True


def _same_nullable(left: Any, right: Any) -> bool:
    if _is_null(left) or _is_null(right):
        return _is_null(left) and _is_null(right)
    return type(left) is type(right) and left == right


def restore_descriptions(
    storage: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    original: pd.DataFrame | None = None,
    null_kinds: tuple[tuple[int, str], ...] = (),
    original_dtype: Any = object,
) -> pd.Series:
    """Restore Description values by row_id after validating manifest consistency."""
    validate_description_manifest(storage, manifest, original=original)
    encoded = _check_storage(storage)
    entries = _check_manifest(manifest)
    if not isinstance(null_kinds, tuple):
        raise TextExceptionError("Null restoration metadata is malformed.")
    null_map: dict[int, str] = {}
    for item in null_kinds:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not int
            or item[0] in null_map
            or item[1] not in {"none", "pd.NA", "nan", "NaT"}
        ):
            raise TextExceptionError("Null restoration metadata is malformed.")
        null_map[item[0]] = item[1]

    storage_ids = set(encoded["row_id"].tolist())
    if not set(null_map) <= storage_ids:
        raise TextExceptionError("Null restoration metadata contains unknown row mappings.")
    null_positions = {
        int(row_id)
        for row_id, value in zip(encoded["row_id"].tolist(), encoded["Description"].tolist())
        if _is_null(value)
    }
    if set(null_map) != null_positions:
        raise TextExceptionError("Null restoration metadata does not match encoded nulls.")
    if original is not None:
        expected_nulls: dict[int, str] = {}
        for row_id, value in zip(original["row_id"].tolist(), original["Description"].tolist()):
            if _is_null(value):
                expected_nulls[int(row_id)] = (
                    "none" if value is None else "pd.NA" if value is pd.NA
                    else "NaT" if value is pd.NaT else "nan"
                )
        if null_map != expected_nulls:
            raise TextExceptionError("Null restoration metadata disagrees with the original frame.")

    entry_by_id = {int(row.row_id): (row.original_type, row.text) for row in entries.itertuples(index=False)}
    values: list[Any] = []
    for row_id, value in zip(encoded["row_id"].tolist(), encoded["Description"].tolist()):
        row_id = int(row_id)
        if row_id in entry_by_id:
            type_tag, text = entry_by_id[row_id]
            values.append(_decode_scalar(type_tag, text))
        elif _is_null(value):
            marker = null_map.get(row_id)
            if marker is None:
                # A null without a sentinel record remains a null, represented
                # by pandas' standard missing marker.
                values.append(pd.NA)
            else:
                values.append({"none": None, "pd.NA": pd.NA, "nan": np.nan, "NaT": pd.NaT}[marker])
        else:
            if row_id in null_map:
                raise TextExceptionError("Null restoration metadata conflicts with encoded data.")
            values.append(value)
    try:
        restored = pd.Series(values, index=encoded.index, name="Description", dtype=original_dtype)
    except (TypeError, ValueError, OverflowError):
        raise TextExceptionError("Description values cannot be restored to the recorded dtype.") from None
    return restored
