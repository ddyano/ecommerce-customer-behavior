"""Synthetic tests for persistence memory preflight and bounded footer checks."""
from __future__ import annotations

import struct

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.persistence.memory_preflight as preflight
from src.persistence.memory_preflight import (
    MemoryEstimate,
    MemoryLimits,
    MemoryPreflightError,
    estimate_registration,
    estimate_verification,
    inspect_parquet_footer,
    require_capacity,
)


def test_registration_estimate_handles_empty_frame():
    estimate = estimate_registration(pd.DataFrame())
    assert estimate.rows == 0
    assert estimate.columns == 0
    assert estimate.estimated_peak_bytes > 0


def test_explicit_byte_limit_accepts_exact_boundary_and_rejects_one_less():
    estimate = MemoryEstimate(1, "synthetic", 100, 10, 1, 1)
    require_capacity(estimate, MemoryLimits(max_estimated_bytes=100))
    with pytest.raises(MemoryPreflightError, match="byte limit"):
        require_capacity(estimate, MemoryLimits(max_estimated_bytes=99))


def test_available_memory_limit_accepts_exact_boundary_and_rejects_excess():
    estimate = MemoryEstimate(1, "synthetic", 50, 10, 1, 1)
    limits = MemoryLimits(max_available_fraction=0.5)
    require_capacity(estimate, limits, available_bytes=100)
    with pytest.raises(MemoryPreflightError, match="available-memory"):
        require_capacity(
            MemoryEstimate(1, "synthetic", 51, 10, 1, 1),
            limits,
            available_bytes=100,
        )


def test_required_available_memory_must_be_obtainable(monkeypatch):
    monkeypatch.setattr(preflight, "available_memory_bytes", lambda: None)
    with pytest.raises(MemoryPreflightError, match="unavailable"):
        require_capacity(
            MemoryEstimate(1, "synthetic", 1, 1, 0, 0),
            MemoryLimits(max_available_fraction=0.5),
        )


@pytest.mark.parametrize(
    "limits",
    [
        MemoryLimits(max_estimated_bytes=-1),
        MemoryLimits(max_available_fraction=0.0),
        MemoryLimits(max_available_fraction=1.1),
    ],
)
def test_invalid_limits_are_rejected(limits):
    with pytest.raises(MemoryPreflightError):
        require_capacity(
            MemoryEstimate(1, "synthetic", 1, 1, 0, 0),
            limits,
            available_bytes=100,
        )


def test_footer_inspection_and_verification_estimate_use_metadata_only(tmp_path):
    path = tmp_path / "small.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3]}), path)
    footer = inspect_parquet_footer(path)
    assert footer.rows == 3
    assert footer.columns == 1
    assert footer.footer_size > 0
    estimate = estimate_verification(footer)
    assert estimate.rows == 3
    assert estimate.estimated_peak_bytes > footer.file_size


def test_footer_size_limit_accepts_exact_boundary_and_rejects_one_less(tmp_path):
    path = tmp_path / "small.parquet"
    pq.write_table(pa.table({"value": [1]}), path)
    measured = inspect_parquet_footer(path)
    assert inspect_parquet_footer(
        path, max_footer_bytes=measured.footer_size
    ) == measured
    with pytest.raises(MemoryPreflightError, match="footer size"):
        inspect_parquet_footer(path, max_footer_bytes=measured.footer_size - 1)


def test_footer_structure_accepts_exact_header_footer_trailer_boundary(
    tmp_path, monkeypatch
):
    footer_size = 20
    path = tmp_path / "boundary.parquet"
    path.write_bytes(
        b"PAR1" + b"x" * footer_size + struct.pack("<I", footer_size) + b"PAR1"
    )

    class Metadata:
        num_rows = 0
        num_columns = 1
        num_row_groups = 0

    class ParquetFile:
        metadata = Metadata()

        def __init__(self, _path):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pq, "ParquetFile", ParquetFile)
    footer = inspect_parquet_footer(path, max_footer_bytes=footer_size)
    assert footer.file_size == footer_size + 12
    assert footer.footer_size == footer_size


def test_footer_structure_rejects_one_byte_short_before_pyarrow(tmp_path, monkeypatch):
    footer_size = 20
    path = tmp_path / "short.parquet"
    path.write_bytes(
        b"PAR1" + b"x" * (footer_size - 1) + struct.pack("<I", footer_size) + b"PAR1"
    )

    def must_not_parse(*args, **kwargs):
        raise AssertionError("PyArrow parsed an impossible footer boundary")

    monkeypatch.setattr(pq, "ParquetFile", must_not_parse)
    with pytest.raises(MemoryPreflightError, match="footer size"):
        inspect_parquet_footer(path, max_footer_bytes=footer_size)


def test_footer_limit_is_enforced_before_pyarrow_parses_it(tmp_path, monkeypatch):
    path = tmp_path / "small.parquet"
    pq.write_table(pa.table({"value": [1]}), path)

    def must_not_parse(*args, **kwargs):
        raise AssertionError("PyArrow parsed an over-limit footer")

    monkeypatch.setattr(pq, "ParquetFile", must_not_parse)
    with pytest.raises(MemoryPreflightError, match="footer size"):
        inspect_parquet_footer(path, max_footer_bytes=1)


@pytest.mark.parametrize("content", [b"", b"PAR1bad!", b"not-parquet-data"])
def test_malformed_footer_is_rejected(tmp_path, content):
    path = tmp_path / "bad.parquet"
    path.write_bytes(content)
    with pytest.raises(MemoryPreflightError):
        inspect_parquet_footer(path)
