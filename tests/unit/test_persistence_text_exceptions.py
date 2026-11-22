"""Synthetic tests for Description exception encoding and restoration."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import CleaningConfig
from src.persistence.text_exceptions import (
    MANIFEST_FIELDS,
    EncodedDescriptions,
    TextExceptionError,
    encode_descriptions,
    restore_descriptions,
    validate_description_manifest,
)
from src.preprocessing.cleaning import clean_transactions


RAW_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
    "Price", "Customer ID", "Country", "source_sheet",
]


@pytest.fixture
def annotated() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            ("1001", "A1", "placeholder", 1, "2020-01-01", 2.0, 1.0, "UK", "S1"),
            ("1002", "A2", "placeholder", 1, "2020-01-02", 2.0, 2.0, "UK", "S1"),
            ("1003", "A3", "placeholder", 1, "2020-01-03", 2.0, 3.0, "FR", "S2"),
            ("1004", "A4", "placeholder", 1, "2020-01-04", 2.0, 4.0, "FR", "S2"),
            ("1005", "A5", "placeholder", 1, "2020-01-05", 2.0, 5.0, "UK", "S1"),
            ("1006", "A6", "placeholder", 1, "2020-01-06", 2.0, 6.0, "UK", "S1"),
            ("1007", "A7", "placeholder", 1, "2020-01-07", 2.0, 7.0, "FR", "S2"),
        ],
        columns=RAW_COLUMNS,
    )
    raw["Description"] = pd.Series(
        ["ordinary", 17, 2.5, True, b"raw-bytes", None, pd.NA], dtype=object
    )
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"]).astype("datetime64[us]")
    raw["Customer ID"] = raw["Customer ID"].astype("float64")
    config = CleaningConfig(
        cancellation_invoice_prefix="C",
        non_product_stock_codes=(),
        duplicate_policy="flag",
        voucher_stock_code_prefixes=(),
    )
    return clean_transactions(raw, config, RAW_COLUMNS[:-1]).annotated


def test_round_trip_preserves_exception_types_strings_nulls_and_dtype(annotated):
    encoded = encode_descriptions(annotated)
    assert tuple(encoded.manifest.columns) == MANIFEST_FIELDS
    assert set(encoded.manifest["original_type"]) == {"int", "float", "bool", "bytes"}
    assert not {"Invoice", "Customer ID"} & set(encoded.manifest.columns)
    assert isinstance(encoded.storage["Description"].dtype, pd.StringDtype)

    restored = restore_descriptions(
        encoded.storage,
        encoded.manifest,
        original=annotated,
        null_kinds=encoded.null_kinds,
        original_dtype=encoded.original_dtype,
    )
    expected = annotated["Description"]
    assert restored.dtype == expected.dtype
    assert [type(x) for x in restored.iloc[:5]] == [type(x) for x in expected.iloc[:5]]
    assert restored.iloc[0] == "ordinary"
    assert restored.iloc[1] == 17
    assert restored.iloc[2] == 2.5
    assert restored.iloc[3] is True
    assert restored.iloc[4] == b"raw-bytes"
    assert restored.iloc[5] is None
    assert restored.iloc[6] is pd.NA


def test_full_coverage_is_verified_only_with_original(annotated):
    encoded = encode_descriptions(annotated)
    assert validate_description_manifest(encoded.storage, encoded.manifest) is False
    assert validate_description_manifest(
        encoded.storage, encoded.manifest, original=annotated
    ) is True


def test_missing_manifest_entry_fails_completeness_but_consistent_mode_is_not_claimed(annotated):
    encoded = encode_descriptions(annotated)
    incomplete = encoded.manifest.iloc[:-1].copy()
    assert validate_description_manifest(encoded.storage, incomplete) is False
    with pytest.raises(TextExceptionError, match="completely"):
        validate_description_manifest(encoded.storage, incomplete, original=annotated)


def test_duplicate_manifest_row_id_is_rejected(annotated):
    encoded = encode_descriptions(annotated)
    duplicate = pd.concat([encoded.manifest, encoded.manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(TextExceptionError, match="duplicate"):
        restore_descriptions(encoded.storage, duplicate)


@pytest.mark.parametrize("tamper", ["text", "source_sheet", "stock_code", "row_id", "type"])
def test_tampered_manifest_is_rejected(annotated, tamper):
    encoded = encode_descriptions(annotated)
    bad = encoded.manifest.copy()
    if tamper == "text":
        bad.loc[0, "text"] = "tampered"
    elif tamper == "source_sheet":
        bad.loc[0, "source_sheet"] = "wrong-sheet"
    elif tamper == "stock_code":
        bad.loc[0, "stock_code"] = "wrong-stock"
    elif tamper == "row_id":
        bad.loc[0, "row_id"] = 99999
    else:
        bad.loc[0, "original_type"] = "object"
    with pytest.raises(TextExceptionError):
        validate_description_manifest(encoded.storage, bad)


def test_manifest_rejects_missing_extra_or_reordered_fields(annotated):
    encoded = encode_descriptions(annotated)
    for bad in (
        encoded.manifest.drop(columns=["stock_code"]),
        encoded.manifest.assign(Invoice="forbidden"),
        encoded.manifest.loc[:, list(reversed(encoded.manifest.columns))],
    ):
        with pytest.raises(TextExceptionError):
            validate_description_manifest(encoded.storage, bad)


def test_ambiguous_storage_row_mapping_is_rejected(annotated):
    encoded = encode_descriptions(annotated)
    duplicate_storage = pd.concat([encoded.storage, encoded.storage.iloc[[0]]], ignore_index=True)
    with pytest.raises(TextExceptionError, match="duplicate"):
        validate_description_manifest(duplicate_storage, encoded.manifest)


def test_unknown_manifest_row_mapping_is_rejected(annotated):
    encoded = encode_descriptions(annotated)
    bad = encoded.manifest.copy()
    bad.loc[0, "row_id"] = 100000
    with pytest.raises(TextExceptionError, match="row mappings"):
        validate_description_manifest(encoded.storage, bad)


def test_storage_data_and_manifest_do_not_expose_forbidden_columns(annotated):
    encoded = encode_descriptions(annotated)
    assert tuple(encoded.storage.columns) == ("row_id", "Description", "source_sheet", "StockCode")
    assert tuple(encoded.manifest.columns) == MANIFEST_FIELDS


def test_errors_do_not_echo_description_or_identifier_values(annotated):
    encoded = encode_descriptions(annotated)
    bad = encoded.manifest.copy()
    secret_description = bad.loc[0, "text"]
    bad.loc[0, "text"] = "wrong"
    with pytest.raises(TextExceptionError) as error:
        validate_description_manifest(encoded.storage, bad)
    message = str(error.value)
    assert secret_description not in message
    assert "1002" not in message
    assert "Customer ID" not in message
    assert "Invoice" not in message


def test_null_metadata_must_match_storage_nulls(annotated):
    encoded = encode_descriptions(annotated)
    wrong_nulls = tuple((row_id, marker) for row_id, marker in encoded.null_kinds if row_id != 5)
    with pytest.raises(TextExceptionError, match="nulls"):
        restore_descriptions(
            encoded.storage,
            encoded.manifest,
            null_kinds=wrong_nulls,
            original_dtype=encoded.original_dtype,
        )


def test_nullable_stock_code_round_trips_in_manifest(annotated):
    changed = annotated.copy()
    changed["StockCode"] = changed["StockCode"].astype("string")
    exception_row = int(annotated.loc[annotated["Description"].map(type).eq(int), "row_id"].iloc[0])
    changed.loc[changed["row_id"] == exception_row, "StockCode"] = pd.NA

    encoded = encode_descriptions(changed)
    entry = encoded.manifest.loc[encoded.manifest["row_id"] == exception_row]
    assert entry["stock_code"].isna().all()
    assert validate_description_manifest(encoded.storage, encoded.manifest, original=changed)
    restored = restore_descriptions(
        encoded.storage,
        encoded.manifest,
        original=changed,
        null_kinds=encoded.null_kinds,
        original_dtype=encoded.original_dtype,
    )
    assert restored.iloc[exception_row] == changed.loc[
        changed["row_id"] == exception_row, "Description"
    ].iloc[0]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra", "conflicting"])
def test_null_kinds_must_cover_each_null_exactly_once(annotated, mode):
    encoded = encode_descriptions(annotated)
    null_kinds = list(encoded.null_kinds)
    if mode == "missing":
        null_kinds = null_kinds[:-1]
    elif mode == "duplicate":
        null_kinds.append(null_kinds[0])
    elif mode == "extra":
        null_kinds.append((0, "none"))
    else:
        null_kinds = [
            (row_id, "pd.NA" if row_id == 5 else marker)
            for row_id, marker in null_kinds
        ]
    with pytest.raises(TextExceptionError):
        restore_descriptions(
            encoded.storage,
            encoded.manifest,
            original=annotated,
            null_kinds=tuple(null_kinds),
            original_dtype=encoded.original_dtype,
        )


@pytest.mark.parametrize("malformation", ["null", "non_string_dtype"])
def test_manifest_referenced_malformed_stored_description_raises_domain_error(
    annotated, malformation
):
    encoded = encode_descriptions(annotated)
    bad_storage = encoded.storage.copy()
    row_id = int(encoded.manifest.iloc[0]["row_id"])
    position = bad_storage.index[bad_storage["row_id"] == row_id][0]
    if malformation == "null":
        bad_storage.loc[position, "Description"] = pd.NA
    else:
        bad_storage["Description"] = bad_storage["Description"].astype(object)
    with pytest.raises(TextExceptionError):
        validate_description_manifest(bad_storage, encoded.manifest)


def test_restoration_maps_nonconsecutive_reordered_rows_by_row_id(annotated):
    changed = annotated.copy()
    changed["row_id"] = pd.Series(
        [41, 7, 103, 22, 88, 5, 61], dtype="int64"
    )
    encoded = encode_descriptions(changed)
    reordered_storage = encoded.storage.iloc[::-1].reset_index(drop=True)

    restored = restore_descriptions(
        reordered_storage,
        encoded.manifest,
        original=changed,
        null_kinds=encoded.null_kinds,
        original_dtype=encoded.original_dtype,
    )
    expected = (
        changed.set_index("row_id")
        .loc[reordered_storage["row_id"].tolist(), "Description"]
        .reset_index(drop=True)
    )
    pd.testing.assert_series_equal(restored, expected)
