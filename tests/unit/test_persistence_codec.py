"""Synthetic tests for the single-file Parquet persistence codec."""
from __future__ import annotations

import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import CleaningConfig
import src.persistence.codec as codec_module
from src.persistence.codec import (
    CodecError,
    decode_parquet,
    encode_parquet,
    verify_parquet,
)
from src.persistence.hashing import canonical_json_bytes
from src.persistence.schema import ANNOTATED_COLUMNS, CATEGORICAL_COLUMNS
from src.preprocessing.cleaning import clean_transactions

METADATA_KEY = b"ecommerce.persistence.codec.v1"
RAW_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
    "Price", "Customer ID", "Country", "source_sheet",
]


@pytest.fixture
def annotated() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            ("1001", "A1", "x", 1, "2020-01-01", 2.0, 1.0, "UK", "S1"),
            ("1002", "A2", "x", 1, "2020-01-02", 2.0, None, "UK", "S1"),
            ("1003", "A3", "x", -1, "2020-01-03", 3.0, 3.0, "FR", "S2"),
            ("1004", "A4", "x", 1, "2020-01-04", 4.0, 4.0, "FR", "S2"),
        ],
        columns=RAW_COLUMNS,
    )
    raw["Description"] = pd.Series(["ordinary", 17, None, pd.NA], dtype=object)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"]).astype("datetime64[us]")
    raw["Customer ID"] = raw["Customer ID"].astype("float64")
    config = CleaningConfig(
        cancellation_invoice_prefix="C",
        non_product_stock_codes=(),
        duplicate_policy="flag",
        voucher_stock_code_prefixes=(),
    )
    return clean_transactions(raw, config, RAW_COLUMNS[:-1]).annotated


def rewrite_with_metadata(source, target, edit):
    table = pq.read_table(source)
    metadata = dict(table.schema.metadata or {})
    payload = json.loads(metadata[METADATA_KEY].decode("utf-8"))
    edit(payload)
    metadata[METADATA_KEY] = canonical_json_bytes(payload)
    pq.write_table(table.replace_schema_metadata(metadata), target)


def test_round_trip_preserves_all_columns_dtypes_categories_and_description(annotated, tmp_path):
    path = tmp_path / "transactions.parquet"
    result = encode_parquet(annotated, path)
    restored = decode_parquet(path)
    pd.testing.assert_frame_equal(restored, annotated)
    assert tuple(restored.columns) == ANNOTATED_COLUMNS
    for column in CATEGORICAL_COLUMNS:
        assert restored[column].dtype == annotated[column].dtype
    assert restored["Description"].iloc[1] == 17
    assert type(restored["Description"].iloc[1]) is int
    assert restored["Description"].iloc[2] is None
    assert restored["Description"].iloc[3] is pd.NA
    assert result == verify_parquet(path)


def test_empty_frame_round_trip(annotated, tmp_path):
    path = tmp_path / "empty.parquet"
    empty = annotated.iloc[:0].copy()
    encode_parquet(empty, path)
    pd.testing.assert_frame_equal(decode_parquet(path), empty)
    payload = json.loads(pq.read_table(path).schema.metadata[METADATA_KEY].decode("utf-8"))
    assert payload["dynamic_categories"] == [["Country", []], ["source_sheet", []]]


def test_empty_artifact_rejects_nonempty_dynamic_vocabulary(annotated, tmp_path):
    source = tmp_path / "empty.parquet"
    corrupted = tmp_path / "tampered-vocabulary.parquet"
    encode_parquet(annotated.iloc[:0].copy(), source)

    def add_country(payload):
        for item in payload["dynamic_categories"]:
            if item[0] == "Country":
                item[1].append("Unexpected")

    rewrite_with_metadata(source, corrupted, add_country)
    with pytest.raises(CodecError):
        decode_parquet(corrupted)


def test_empty_artifact_rejects_unexpected_categorical_physical_type(annotated, tmp_path):
    source = tmp_path / "empty.parquet"
    corrupted = tmp_path / "tampered-type.parquet"
    encode_parquet(annotated.iloc[:0].copy(), source)
    table = pq.read_table(source)
    index = table.schema.get_field_index("Country")
    field = pa.field("Country", pa.int64(), nullable=True)
    changed = table.set_column(index, field, pa.array([], type=pa.int64()))
    changed = changed.replace_schema_metadata(table.schema.metadata)
    pq.write_table(changed, corrupted)

    with pytest.raises(CodecError):
        decode_parquet(corrupted)


@pytest.mark.parametrize(
    "edit",
    [
        lambda meta: meta.update(schema_version=999),
        lambda meta: meta.update(schema_digest="0" * 64),
        lambda meta: meta.update(exception_manifest=[]),
        lambda meta: meta["fingerprint"].update(digest="0" * 64),
    ],
    ids=["schema-version", "schema-digest", "incomplete-manifest", "fingerprint"],
)
def test_metadata_tampering_is_rejected(annotated, tmp_path, edit):
    source = tmp_path / "source.parquet"
    corrupted = tmp_path / "corrupted.parquet"
    encode_parquet(annotated, source)
    rewrite_with_metadata(source, corrupted, edit)
    with pytest.raises(CodecError):
        decode_parquet(corrupted)


@pytest.mark.parametrize("tamper", ["dtype", "dynamic_categories"])
def test_malformed_dtype_or_category_metadata_raises_safe_codec_error(
    annotated, tmp_path, tamper
):
    source = tmp_path / "source.parquet"
    corrupted = tmp_path / "corrupted.parquet"
    encode_parquet(annotated, source)

    def edit(payload):
        if tamper == "dtype":
            payload["original_dtypes"][1][1] = {"kind": "unsupported", "name": "secret"}
        else:
            for item in payload["dynamic_categories"]:
                if item[0] == "Country":
                    item[1].append("unexpected-country")

    rewrite_with_metadata(source, corrupted, edit)
    with pytest.raises(CodecError) as caught:
        decode_parquet(corrupted)
    assert "secret" not in str(caught.value)
    assert "unexpected-country" not in str(caught.value)


def test_parquet_data_change_without_fingerprint_update_is_rejected(annotated, tmp_path):
    source = tmp_path / "source.parquet"
    corrupted = tmp_path / "corrupted-data.parquet"
    encode_parquet(annotated, source)
    table = pq.read_table(source)
    index = table.schema.get_field_index("Price")
    field = table.schema.field(index)
    values = table.column(index).to_pylist()
    values[0] += 1.0
    changed = table.set_column(index, field, pa.array(values, type=field.type))
    pq.write_table(changed, corrupted)

    with pytest.raises(CodecError, match="fingerprint"):
        decode_parquet(corrupted)


def test_noncanonical_or_missing_metadata_is_rejected(annotated, tmp_path):
    source = tmp_path / "source.parquet"
    missing = tmp_path / "missing.parquet"
    encode_parquet(annotated, source)
    table = pq.read_table(source)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(METADATA_KEY)
    pq.write_table(table.replace_schema_metadata(metadata), missing)
    with pytest.raises(CodecError):
        verify_parquet(missing)


def test_failed_decode_releases_parquet_file_handle(annotated, tmp_path):
    source = tmp_path / "source.parquet"
    corrupted = tmp_path / "corrupted.parquet"
    encode_parquet(annotated, source)
    rewrite_with_metadata(
        source,
        corrupted,
        lambda payload: payload.update(schema_version=999),
    )
    with pytest.raises(CodecError):
        decode_parquet(corrupted)
    corrupted.unlink()
    assert not corrupted.exists()


def test_corrupt_file_returns_safe_codec_error(tmp_path):
    path = tmp_path / "corrupt.parquet"
    path.write_bytes(b"not parquet; Customer ID=secret")
    with pytest.raises(CodecError) as caught:
        decode_parquet(path)
    assert "secret" not in str(caught.value)


def test_exception_manifest_metadata_has_only_approved_fields_and_no_identifiers(
    annotated, tmp_path
):
    path = tmp_path / "transactions.parquet"
    encode_parquet(annotated, path)
    table = pq.read_table(path)
    payload = json.loads(table.schema.metadata[METADATA_KEY].decode("utf-8"))
    records = payload["exception_manifest"]
    assert all(
        set(record) == {"row_id", "original_type", "text", "source_sheet", "stock_code"}
        for record in records
    )
    forbidden_fields = {"Invoice", "Customer ID"}
    assert all(forbidden_fields.isdisjoint(record) for record in records)
    serialized = table.schema.metadata[METADATA_KEY].decode("utf-8")
    assert "1001" not in serialized


def test_write_failure_preserves_existing_destination_and_cleans_temp(
    annotated, tmp_path, monkeypatch
):
    path = tmp_path / "existing.parquet"
    path.write_bytes(b"previous artifact")
    original = path.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("sensitive path and transaction data")

    monkeypatch.setattr(pq, "write_table", fail_write)
    with pytest.raises(CodecError) as caught:
        encode_parquet(annotated, path)
    assert str(caught.value) == "Parquet encoding failed; destination was not replaced."
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".existing.parquet.*.tmp")) == []


def test_fsync_failure_preserves_existing_destination_and_cleans_temp(
    annotated, tmp_path, monkeypatch
):
    path = tmp_path / "existing.parquet"
    path.write_bytes(b"previous artifact")
    original = path.read_bytes()

    def fail_fsync(_descriptor):
        raise OSError("sensitive filesystem detail")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(CodecError) as caught:
        encode_parquet(annotated, path)
    assert str(caught.value) == "Parquet encoding failed; destination was not replaced."
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".existing.parquet.*.tmp")) == []


def test_bad_target_replacement_keeps_destination_and_cleans_temp(
    annotated, tmp_path, monkeypatch
):
    path = tmp_path / "existing.parquet"
    path.write_bytes(b"previous artifact")
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("replace denied")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(CodecError):
        encode_parquet(annotated, path)
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".existing.parquet.*.tmp")) == []


def test_internal_decode_failure_preserves_destination_and_cleans_temp(
    annotated, tmp_path, monkeypatch
):
    path = tmp_path / "existing.parquet"
    path.write_bytes(b"previous artifact")
    original = path.read_bytes()

    def fail_decode(_path):
        raise CodecError("synthetic decode failure")

    monkeypatch.setattr(codec_module, "_decode_path", fail_decode)
    with pytest.raises(CodecError, match="synthetic decode failure"):
        encode_parquet(annotated, path)
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".existing.parquet.*.tmp")) == []


def test_wrong_parquet_column_order_is_rejected(annotated, tmp_path):
    source = tmp_path / "source.parquet"
    bad = tmp_path / "bad-order.parquet"
    encode_parquet(annotated, source)
    table = pq.read_table(source).select(list(reversed(ANNOTATED_COLUMNS)))
    # Keep metadata so the structure check, not the missing-metadata path, fires.
    metadata = pq.read_schema(source).metadata
    pq.write_table(table.replace_schema_metadata(metadata), bad)
    with pytest.raises(CodecError):
        decode_parquet(bad)
