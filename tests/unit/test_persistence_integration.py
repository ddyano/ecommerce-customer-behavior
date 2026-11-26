"""Synthetic end-to-end tests for the persistence workflow.

These tests intentionally use small generated frames.  They do not read the
project's transaction datasets.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import src.persistence.codec as codec_module
import src.persistence.expectations as expectations_module
from src.config import CleaningConfig
from src.persistence.codec import CodecError, decode_parquet, encode_parquet, verify_parquet
from src.persistence.expectations import (
    ExpectationConflictError,
    ExpectationsError,
    register_expectation,
    verify_expected_artifact,
)
from src.persistence.hashing import canonical_json_bytes
from src.persistence.memory_preflight import MemoryLimits, MemoryPreflightError
from src.persistence.schema import (
    ANNOTATED_COLUMNS,
    CATEGORICAL_COLUMNS,
    DYNAMIC_CATEGORICALS,
    RULE_CATEGORICALS,
    SchemaError,
    to_persistence_view,
    validate_annotated_view,
)
from src.persistence.text_exceptions import MANIFEST_FIELDS
from src.preprocessing.cleaning import clean_transactions

METADATA_KEY = b"ecommerce.persistence.codec.v1"
RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "source_sheet",
]


@pytest.fixture
def annotated() -> pd.DataFrame:
    descriptions = pd.Series(
        [
            "DESCRIPTION-PRIVATE-ALPHA",
            17,
            2.5,
            True,
            b"synthetic-bytes",
            None,
            pd.NA,
            float("nan"),
            "ordinary-secondary",
        ],
        dtype=object,
    )
    raw = pd.DataFrame(
        {
            "Invoice": [f"INVOICE-PRIVATE-{index}" for index in range(9)],
            "StockCode": [f"SKU{index}" for index in range(9)],
            "Description": descriptions,
            "Quantity": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "InvoiceDate": pd.date_range("2020-01-01", periods=9).astype(
                "datetime64[us]"
            ),
            "Price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Customer ID": [900001.0, None, 900003.0, 900004.0, None, 900006.0,
                            900007.0, 900008.0, 900009.0],
            "Country": ["UK", "FR", None, "UK", "FR", "UK", "FR", "UK", "FR"],
            "source_sheet": ["S1", "S1", "S1", "S2", "S2", "S2", "S1", "S2", "S1"],
        },
        columns=RAW_COLUMNS,
    )
    config = CleaningConfig(
        cancellation_invoice_prefix="C",
        non_product_stock_codes=(),
        duplicate_policy="flag",
        voucher_stock_code_prefixes=(),
    )
    frame = clean_transactions(raw, config, RAW_COLUMNS[:-1]).annotated

    # Exercise a nullable string identifier in an exceptional Description row.
    frame["StockCode"] = frame["StockCode"].astype("string")
    frame.loc[frame["Description"].map(type).eq(int), "StockCode"] = pd.NA

    # Persistence supports unique, nonnegative row IDs independently of position.
    frame["row_id"] = pd.Series(
        [101, 7, 305, 44, 902, 18, 63, 501, 29], dtype="int64"
    )
    frame = frame.iloc[[8, 2, 6, 0, 4, 1, 7, 3, 5]].reset_index(drop=True)
    return frame


def _registry(tmp_path):
    path = tmp_path / "expectations"
    path.mkdir()
    return path


def _metadata(path) -> dict:
    table = pq.read_table(path)
    return json.loads(table.schema.metadata[METADATA_KEY].decode("utf-8"))


def _rewrite_metadata(source, target, edit) -> None:
    table = pq.read_table(source)
    metadata = dict(table.schema.metadata or {})
    payload = json.loads(metadata[METADATA_KEY].decode("utf-8"))
    edit(payload)
    metadata[METADATA_KEY] = canonical_json_bytes(payload)
    pq.write_table(table.replace_schema_metadata(metadata), target)


def _assert_description_identities(actual: pd.Series, expected: pd.Series) -> None:
    for restored, original in zip(actual.tolist(), expected.tolist()):
        if original is None:
            assert restored is None
        elif original is pd.NA:
            assert restored is pd.NA
        elif type(original) is float and math.isnan(original):
            assert type(restored) is float and math.isnan(restored)
        else:
            assert type(restored) is type(original)
            assert restored == original


def _assert_exact_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=True,
        check_categorical=True,
    )
    assert tuple(actual.columns) == ANNOTATED_COLUMNS
    _assert_description_identities(actual["Description"], expected["Description"])


def test_synthetic_fixture_preserves_supported_types_after_cleaning(annotated):
    validate_annotated_view(annotated)
    assert tuple(annotated.columns) == ANNOTATED_COLUMNS
    assert annotated["row_id"].tolist() == [29, 305, 63, 101, 902, 7, 501, 44, 18]
    assert annotated["StockCode"].dtype == pd.StringDtype()
    assert annotated["StockCode"].isna().sum() == 1
    assert annotated["Customer ID"].dtype == np.dtype("float64")
    assert annotated["Customer ID"].isna().sum() == 2
    expected_types = {str, int, float, bool, bytes, type(None), type(pd.NA)}
    assert {type(value) for value in annotated["Description"].tolist()} == expected_types
    _assert_description_identities(annotated["Description"], annotated["Description"])


def test_schema_rejects_nat_in_object_description(annotated):
    invalid = annotated.copy(deep=True)
    assert invalid["Description"].dtype == np.dtype(object)
    invalid.loc[invalid.index[0], "Description"] = pd.NaT

    with pytest.raises(SchemaError, match="Description"):
        validate_annotated_view(invalid)


def test_complete_stable_file_workflow_round_trips_exactly(annotated, tmp_path):
    registry = _registry(tmp_path)
    artifact = tmp_path / "transactions.parquet"
    register_expectation(registry, "transactions", annotated)
    encoded = encode_parquet(annotated, artifact)
    verified = verify_expected_artifact(registry, "transactions", artifact)

    # This subsequent decode is honest only for a stable path: the two public
    # calls do not protect against replacement occurring between them.
    restored = decode_parquet(artifact)

    assert encoded == verified == verify_parquet(artifact)
    _assert_exact_frame(restored, annotated)

    expected_view = to_persistence_view(annotated)
    restored_view = to_persistence_view(restored)
    assert restored_view.dynamic_categories == expected_view.dynamic_categories
    for column in CATEGORICAL_COLUMNS:
        assert restored_view.frame[column].dtype == expected_view.frame[column].dtype
        assert tuple(restored_view.frame[column].cat.categories) == tuple(
            expected_view.frame[column].cat.categories
        )
    for column, vocabulary in RULE_CATEGORICALS.items():
        assert tuple(restored_view.frame[column].cat.categories) == vocabulary
    assert dict(restored_view.dynamic_categories) == {
        "Country": ("FR", "UK"),
        "source_sheet": ("S1", "S2"),
    }


def test_complete_empty_frame_workflow_round_trips_exactly(annotated, tmp_path):
    empty = annotated.iloc[:0].copy()
    registry = _registry(tmp_path)
    artifact = tmp_path / "empty.parquet"
    register_expectation(registry, "empty", empty)
    encode_parquet(empty, artifact)
    verify_expected_artifact(registry, "empty", artifact)
    restored = decode_parquet(artifact)

    _assert_exact_frame(restored, empty)
    restored_view = to_persistence_view(restored)
    assert dict(restored_view.dynamic_categories) == {
        name: () for name in DYNAMIC_CATEGORICALS
    }
    for column, vocabulary in RULE_CATEGORICALS.items():
        assert tuple(restored_view.frame[column].cat.categories) == vocabulary


def test_valid_different_artifact_fails_independent_verification(annotated, tmp_path):
    registry = _registry(tmp_path)
    artifact = tmp_path / "different.parquet"
    different = annotated.copy(deep=True)
    different.loc[different.index[0], "Price"] += 1.0
    register_expectation(registry, "trusted", annotated)
    encode_parquet(different, artifact)

    assert verify_parquet(artifact).rows == len(different)
    with pytest.raises(ExpectationsError, match="independent expectation"):
        verify_expected_artifact(registry, "trusted", artifact)


@pytest.mark.parametrize(
    "tamper",
    ["manifest", "dtype", "category", "schema-version"],
)
def test_metadata_tampering_fails_independent_and_codec_validation(
    annotated, tmp_path, tamper
):
    registry = _registry(tmp_path)
    source = tmp_path / "source.parquet"
    changed = tmp_path / f"{tamper}.parquet"
    register_expectation(registry, "trusted", annotated)
    encode_parquet(annotated, source)

    def edit(payload):
        if tamper == "manifest":
            payload["exception_manifest"][0]["text"] = "tampered"
        elif tamper == "dtype":
            payload["original_dtypes"][2][1] = {
                "kind": "unsupported",
                "name": "PRIVATE-DTYPE-VALUE",
            }
        elif tamper == "category":
            for item in payload["dynamic_categories"]:
                if item[0] == "Country":
                    item[1].append("ZZ")
        else:
            payload["schema_version"] = 999

    _rewrite_metadata(source, changed, edit)
    with pytest.raises(ExpectationsError):
        verify_expected_artifact(registry, "trusted", changed)
    with pytest.raises(CodecError):
        decode_parquet(changed)


@pytest.mark.parametrize("kind", ["truncated", "corrupt"])
def test_structurally_invalid_artifacts_fail_closed_and_privately(
    annotated, tmp_path, kind
):
    registry = _registry(tmp_path)
    source = tmp_path / "source.parquet"
    invalid = tmp_path / f"{kind}.parquet"
    register_expectation(registry, "trusted", annotated)
    encode_parquet(annotated, source)
    if kind == "truncated":
        invalid.write_bytes(source.read_bytes()[:-8])
    else:
        invalid.write_bytes(
            b"not parquet; INVOICE-PRIVATE-0; 900001.0; DESCRIPTION-PRIVATE-ALPHA"
        )

    with pytest.raises(MemoryPreflightError) as expected_error:
        verify_expected_artifact(registry, "trusted", invalid)
    with pytest.raises(CodecError) as codec_error:
        decode_parquet(invalid)
    for error in (expected_error.value, codec_error.value):
        message = str(error)
        assert "INVOICE-PRIVATE-0" not in message
        assert "900001.0" not in message
        assert "DESCRIPTION-PRIVATE-ALPHA" not in message


def test_memory_rejection_precedes_full_codec_decode_and_cleans_snapshot(
    annotated, tmp_path, monkeypatch
):
    registry = _registry(tmp_path)
    artifact = tmp_path / "transactions.parquet"
    register_expectation(registry, "trusted", annotated)
    encode_parquet(annotated, artifact)
    called = False

    def unexpected_verify(_path):
        nonlocal called
        called = True
        raise AssertionError("full codec verification must not run")

    monkeypatch.setattr(expectations_module, "verify_parquet", unexpected_verify)
    with pytest.raises(MemoryPreflightError, match="memory"):
        verify_expected_artifact(
            registry,
            "trusted",
            artifact,
            limits=MemoryLimits(max_estimated_bytes=1),
        )
    assert called is False
    assert list(tmp_path.glob(".transactions.parquet.*.verify")) == []


def test_snapshot_size_rejection_precedes_copy_and_leaves_no_snapshot(
    annotated, tmp_path
):
    registry = _registry(tmp_path)
    artifact = tmp_path / "transactions.parquet"
    register_expectation(registry, "trusted", annotated)
    encode_parquet(annotated, artifact)

    with pytest.raises(ExpectationsError, match="snapshot size"):
        verify_expected_artifact(
            registry,
            "trusted",
            artifact,
            max_snapshot_bytes=artifact.stat().st_size - 1,
        )
    assert list(tmp_path.glob(".transactions.parquet.*.verify")) == []


def test_atomic_write_failure_preserves_destination_and_cleans_temporary(
    annotated, tmp_path, monkeypatch
):
    destination = tmp_path / "existing.parquet"
    destination.write_bytes(b"existing artifact")
    original = destination.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("INVOICE-PRIVATE-0; 900001.0; DESCRIPTION-PRIVATE-ALPHA")

    monkeypatch.setattr(codec_module.os, "replace", fail_replace)
    with pytest.raises(CodecError) as caught:
        encode_parquet(annotated, destination)
    assert destination.read_bytes() == original
    assert list(tmp_path.glob(".existing.parquet.*.tmp")) == []
    assert "INVOICE-PRIVATE-0" not in str(caught.value)
    assert "900001.0" not in str(caught.value)
    assert "DESCRIPTION-PRIVATE-ALPHA" not in str(caught.value)


def test_conflicting_registration_preserves_original_record(annotated, tmp_path):
    registry = _registry(tmp_path)
    register_expectation(registry, "trusted", annotated)
    record = next(registry.glob("*.json"))
    original = record.read_bytes()
    different = annotated.copy(deep=True)
    different.loc[different.index[0], "Price"] += 1.0

    with pytest.raises(ExpectationConflictError):
        register_expectation(registry, "trusted", different)
    assert record.read_bytes() == original


def test_exception_manifest_has_exact_allowlist_without_identifier_fields(
    annotated, tmp_path
):
    artifact = tmp_path / "transactions.parquet"
    encode_parquet(annotated, artifact)
    payload = _metadata(artifact)
    manifest = payload["exception_manifest"]

    assert manifest
    assert all(set(record) == set(MANIFEST_FIELDS) for record in manifest)
    assert all("Invoice" not in record and "Customer ID" not in record for record in manifest)
    serialized = json.dumps(manifest, sort_keys=True)
    assert "INVOICE-PRIVATE-" not in serialized
    assert "900001.0" not in serialized
