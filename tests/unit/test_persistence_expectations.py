"""Synthetic tests for independent persistence expectations."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

import src.persistence.expectations as expectations_module
from src.config import CleaningConfig
from src.persistence.codec import CodecError, encode_parquet, verify_parquet
from src.persistence.expectations import (
    ExpectationConflictError,
    ExpectationsError,
    build_expectation,
    lookup_expectation,
    register_expectation,
    verify_expected_artifact,
)
from src.persistence.hashing import canonical_json_bytes
from src.persistence.memory_preflight import MemoryLimits, MemoryPreflightError
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
            ("1002", "A2", "x", 1, "2020-01-02", 3.0, None, "FR", "S2"),
            ("1003", "A3", "x", -1, "2020-01-03", 4.0, 3.0, "UK", "S1"),
        ],
        columns=RAW_COLUMNS,
    )
    raw["Invoice"] = raw["Invoice"].astype("string")
    raw["StockCode"] = raw["StockCode"].astype("string")
    raw["Description"] = pd.Series(["ordinary", 17, None], dtype=object)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"]).astype("datetime64[us]")
    raw["Customer ID"] = raw["Customer ID"].astype("float64")
    config = CleaningConfig(
        cancellation_invoice_prefix="C",
        non_product_stock_codes=(),
        duplicate_policy="flag",
        voucher_stock_code_prefixes=(),
    )
    return clean_transactions(raw, config, RAW_COLUMNS[:-1]).annotated


def registry(tmp_path):
    path = tmp_path / "registry"
    path.mkdir()
    return path


def rewrite_metadata(source, target, edit):
    table = pq.read_table(source)
    metadata = dict(table.schema.metadata or {})
    payload = json.loads(metadata[METADATA_KEY].decode("utf-8"))
    edit(payload)
    metadata[METADATA_KEY] = canonical_json_bytes(payload)
    pq.write_table(table.replace_schema_metadata(metadata), target)


def test_expectation_is_deterministic_and_contains_only_summaries(annotated):
    first = build_expectation(annotated)
    second = build_expectation(annotated.copy())
    assert first == second
    record = first.to_record()
    assert record["manifest_count"] == 1
    assert len(record["manifest_digest"]) == 64
    assert len(record["restoration_digest"]) == 64
    serialized = canonical_json_bytes(record)
    assert b"ordinary" not in serialized
    assert b"1001" not in serialized


def test_registration_uses_hashed_key_and_is_idempotent(annotated, tmp_path):
    root = registry(tmp_path)
    key = "approved-artifact"
    first = register_expectation(root, key, annotated)
    second = register_expectation(root, key, annotated.copy())
    assert first == second
    files = list(root.glob("*.json"))
    assert [file.name for file in files] == [
        hashlib.sha256(key.encode("ascii")).hexdigest() + ".json"
    ]
    assert key not in files[0].name


@pytest.mark.parametrize("key", ["", "../escape", "has space", "é", "a" * 129])
def test_unsafe_registry_keys_are_rejected(annotated, tmp_path, key):
    with pytest.raises(ExpectationsError, match="key"):
        register_expectation(registry(tmp_path), key, annotated)


def test_conflicting_registration_never_overwrites_existing_record(annotated, tmp_path):
    root = registry(tmp_path)
    register_expectation(root, "artifact", annotated)
    record_path = next(root.glob("*.json"))
    before = record_path.read_bytes()
    changed = annotated.copy()
    changed.loc[0, "Price"] += 1.0
    with pytest.raises(ExpectationConflictError):
        register_expectation(root, "artifact", changed)
    assert record_path.read_bytes() == before


def test_missing_and_corrupt_registry_records_are_rejected(annotated, tmp_path):
    root = registry(tmp_path)
    with pytest.raises(ExpectationsError, match="missing"):
        lookup_expectation(root, "missing")
    register_expectation(root, "artifact", annotated)
    record_path = next(root.glob("*.json"))
    record_path.write_bytes(b'{"record":{}}')
    with pytest.raises(ExpectationsError):
        lookup_expectation(root, "artifact")


def test_registry_record_read_is_bounded(annotated, tmp_path):
    root = registry(tmp_path)
    register_expectation(root, "artifact", annotated)
    with pytest.raises(ExpectationsError, match="size"):
        lookup_expectation(root, "artifact", max_record_bytes=8)


def test_restoration_digest_detects_dtype_metadata_changes_with_same_logical_values(
    annotated,
):
    changed = annotated.copy()
    changed["Invoice"] = changed["Invoice"].astype(object)
    first = build_expectation(annotated)
    second = build_expectation(changed)
    assert first.fingerprint == second.fingerprint
    assert first.restoration_digest != second.restoration_digest


def test_manifest_digest_and_count_come_from_description_exceptions(annotated):
    first = build_expectation(annotated)
    changed = annotated.copy()
    changed.loc[1, "Description"] = "17"
    second = build_expectation(changed)
    assert first.manifest_count == 1
    assert second.manifest_count == 0
    assert first.manifest_digest != second.manifest_digest


def test_registered_expectation_verifies_artifact(annotated, tmp_path):
    root = registry(tmp_path)
    artifact = tmp_path / "artifact.parquet"
    register_expectation(root, "artifact", annotated)
    encode_parquet(annotated, artifact)
    result = verify_expected_artifact(root, "artifact", artifact)
    assert result.rows == len(annotated)


def test_valid_different_artifact_fails_independent_expectation(annotated, tmp_path):
    root = registry(tmp_path)
    artifact_b = tmp_path / "artifact-b.parquet"
    frame_b = annotated.copy()
    frame_b.loc[0, "Description"] = "different synthetic description"
    register_expectation(root, "frame-a", annotated)
    encode_parquet(frame_b, artifact_b)
    assert verify_parquet(artifact_b).rows == len(frame_b)
    with pytest.raises(ExpectationsError, match="independent expectation"):
        verify_expected_artifact(root, "frame-a", artifact_b)


@pytest.mark.parametrize("tamper", ["manifest", "restoration"])
def test_independent_metadata_digests_reject_tampering(annotated, tmp_path, tamper):
    root = registry(tmp_path)
    source = tmp_path / "source.parquet"
    changed = tmp_path / "changed.parquet"
    register_expectation(root, "artifact", annotated)
    encode_parquet(annotated, source)

    def edit(payload):
        if tamper == "manifest":
            payload["exception_manifest"] = []
        else:
            payload["description_null_kinds"] = []

    rewrite_metadata(source, changed, edit)
    with pytest.raises(ExpectationsError, match="independent expectation"):
        verify_expected_artifact(root, "artifact", changed)


def test_memory_rejection_occurs_before_full_codec_read(
    annotated, tmp_path, monkeypatch
):
    root = registry(tmp_path)
    artifact = tmp_path / "artifact.parquet"
    register_expectation(root, "artifact", annotated)
    encode_parquet(annotated, artifact)
    called = False

    def must_not_verify(_path):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(expectations_module, "verify_parquet", must_not_verify)
    with pytest.raises(MemoryPreflightError):
        verify_expected_artifact(
            root,
            "artifact",
            artifact,
            limits=MemoryLimits(max_estimated_bytes=0),
        )
    assert not called


def test_registration_memory_rejection_precedes_expectation_construction(
    annotated, tmp_path, monkeypatch
):
    root = registry(tmp_path)
    called = False

    def must_not_convert(_frame):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(expectations_module, "to_persistence_view", must_not_convert)
    with pytest.raises(MemoryPreflightError):
        register_expectation(
            root,
            "artifact",
            annotated,
            limits=MemoryLimits(max_estimated_bytes=0),
        )
    assert not called
    assert list(root.iterdir()) == []


def test_failed_full_verification_cleans_snapshot(annotated, tmp_path, monkeypatch):
    root = registry(tmp_path)
    artifact = tmp_path / "artifact.parquet"
    register_expectation(root, "artifact", annotated)
    encode_parquet(annotated, artifact)

    def fail_verify(_path):
        raise CodecError("synthetic failure")

    monkeypatch.setattr(expectations_module, "verify_parquet", fail_verify)
    with pytest.raises(ExpectationsError, match="full codec"):
        verify_expected_artifact(root, "artifact", artifact)
    assert list(tmp_path.glob(".artifact.parquet.*.verify")) == []


def test_snapshot_copy_failure_is_safe_and_cleans_partial_file(
    annotated, tmp_path, monkeypatch
):
    root = registry(tmp_path)
    artifact = tmp_path / "artifact.parquet"
    register_expectation(root, "artifact", annotated)
    encode_parquet(annotated, artifact)

    def fail_fsync(_descriptor):
        raise OSError("private disk detail")

    monkeypatch.setattr(expectations_module.os, "fsync", fail_fsync)
    with pytest.raises(ExpectationsError, match="snapshot"):
        verify_expected_artifact(root, "artifact", artifact)
    assert list(tmp_path.glob(".artifact.parquet.*.verify")) == []


def test_snapshot_size_limit_rejects_before_copy_and_leaves_no_temporary(
    annotated, tmp_path
):
    root = registry(tmp_path)
    artifact = tmp_path / "artifact.parquet"
    register_expectation(root, "artifact", annotated)
    encode_parquet(annotated, artifact)
    with pytest.raises(ExpectationsError, match="snapshot size"):
        verify_expected_artifact(
            root,
            "artifact",
            artifact,
            max_snapshot_bytes=artifact.stat().st_size - 1,
        )
    assert list(tmp_path.glob(".artifact.parquet.*.verify")) == []
