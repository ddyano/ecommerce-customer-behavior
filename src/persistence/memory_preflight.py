"""Conservative memory estimates for persistence registration and verification."""
from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PREFLIGHT_VERSION = 1
DEFAULT_MAX_FOOTER_BYTES = 16 * 1024 * 1024
FIXED_OVERHEAD_BYTES = 64 * 1024 * 1024


class MemoryPreflightError(RuntimeError):
    """An operation cannot proceed within its configured memory limits."""


@dataclass(frozen=True)
class MemoryLimits:
    max_estimated_bytes: int | None = None
    max_available_fraction: float | None = None


@dataclass(frozen=True)
class MemoryEstimate:
    version: int
    operation: str
    estimated_peak_bytes: int
    input_bytes: int
    rows: int
    columns: int


@dataclass(frozen=True)
class ParquetFooterInfo:
    file_size: int
    footer_size: int
    rows: int
    columns: int
    row_groups: int
    uncompressed_bytes: int


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise MemoryPreflightError(f"{label} must be a valid byte count.")
    return value


def available_memory_bytes() -> int | None:
    """Return currently available physical memory using standard-library APIs."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError):
            return None
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        if type(page_size) is int and type(available_pages) is int:
            return page_size * available_pages
    except (AttributeError, OSError, ValueError):
        pass
    return None


def estimate_registration(frame: pd.DataFrame) -> MemoryEstimate:
    """Estimate peak additional memory before expectation construction."""
    if not isinstance(frame, pd.DataFrame):
        raise MemoryPreflightError("Registration input must be a DataFrame.")
    try:
        frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    except Exception:
        raise MemoryPreflightError("Registration input memory cannot be estimated safely.") from None
    rows = int(len(frame))
    columns = int(len(frame.columns))
    estimated = FIXED_OVERHEAD_BYTES + 4 * frame_bytes + rows * 128 + columns * 4096
    return MemoryEstimate(
        PREFLIGHT_VERSION,
        "expectation_registration",
        estimated,
        frame_bytes,
        rows,
        columns,
    )


def inspect_parquet_footer(
    path: Path | str,
    *,
    max_footer_bytes: int = DEFAULT_MAX_FOOTER_BYTES,
) -> ParquetFooterInfo:
    """Validate the bounded footer before allowing PyArrow to parse it."""
    maximum = _positive_int(max_footer_bytes, "max_footer_bytes")
    artifact = Path(path)
    try:
        file_size = artifact.stat().st_size
        if file_size < 12:
            raise MemoryPreflightError("Parquet artifact structure is malformed.")
        with artifact.open("rb") as stream:
            if stream.read(4) != b"PAR1":
                raise MemoryPreflightError("Parquet artifact structure is malformed.")
            stream.seek(-8, os.SEEK_END)
            trailer = stream.read(8)
        if len(trailer) != 8 or trailer[4:] != b"PAR1":
            raise MemoryPreflightError("Parquet artifact structure is malformed.")
        footer_size = struct.unpack("<I", trailer[:4])[0]
        if footer_size <= 0 or footer_size > maximum or footer_size + 12 > file_size:
            raise MemoryPreflightError("Parquet footer size is invalid or exceeds its limit.")
    except MemoryPreflightError:
        raise
    except OSError:
        raise MemoryPreflightError("Parquet artifact cannot be inspected safely.") from None

    parquet = None
    try:
        parquet = pq.ParquetFile(artifact)
        metadata = parquet.metadata
        rows = int(metadata.num_rows)
        columns = int(metadata.num_columns)
        row_groups = int(metadata.num_row_groups)
        if rows < 0 or columns <= 0 or row_groups < 0 or columns > 10_000:
            raise MemoryPreflightError("Parquet footer contains implausible structure.")
        uncompressed = 0
        for index in range(row_groups):
            size = int(metadata.row_group(index).total_byte_size)
            if size < 0:
                raise MemoryPreflightError("Parquet footer contains an invalid size.")
            uncompressed += size
    except MemoryPreflightError:
        raise
    except Exception:
        raise MemoryPreflightError("Parquet footer metadata is malformed.") from None
    finally:
        if parquet is not None:
            parquet.close()
    return ParquetFooterInfo(
        int(file_size), footer_size, rows, columns, row_groups, uncompressed
    )


def estimate_verification(footer: ParquetFooterInfo) -> MemoryEstimate:
    """Estimate peak decode, restoration, and fingerprint memory from the footer."""
    if not isinstance(footer, ParquetFooterInfo):
        raise MemoryPreflightError("Parquet footer estimate input is malformed.")
    if (
        footer.file_size < 0
        or footer.footer_size <= 0
        or footer.rows < 0
        or footer.columns <= 0
        or footer.row_groups < 0
        or footer.uncompressed_bytes < 0
    ):
        raise MemoryPreflightError("Parquet footer estimate values are malformed.")
    base = max(footer.file_size, footer.uncompressed_bytes)
    estimated = (
        FIXED_OVERHEAD_BYTES
        + 4 * base
        + footer.rows * 128
        + footer.columns * 1024 * 1024
    )
    return MemoryEstimate(
        PREFLIGHT_VERSION,
        "artifact_verification",
        estimated,
        footer.file_size,
        footer.rows,
        footer.columns,
    )


def require_capacity(
    estimate: MemoryEstimate,
    limits: MemoryLimits,
    *,
    available_bytes: int | None = None,
) -> None:
    """Reject an estimate exceeding an explicit or available-memory limit."""
    if not isinstance(estimate, MemoryEstimate) or not isinstance(limits, MemoryLimits):
        raise MemoryPreflightError("Memory preflight inputs are malformed.")
    need = _positive_int(
        estimate.estimated_peak_bytes, "estimated_peak_bytes", allow_zero=True
    )
    if limits.max_estimated_bytes is not None:
        maximum = _positive_int(
            limits.max_estimated_bytes, "max_estimated_bytes", allow_zero=True
        )
        if need > maximum:
            raise MemoryPreflightError("Estimated memory exceeds the configured byte limit.")
    fraction = limits.max_available_fraction
    if fraction is not None:
        if type(fraction) is not float or not 0 < fraction <= 1:
            raise MemoryPreflightError("Available-memory fraction is invalid.")
        available = available_bytes
        if available is None:
            available = available_memory_bytes()
        if available is None:
            raise MemoryPreflightError("Available memory is required but unavailable.")
        available = _positive_int(available, "available_bytes", allow_zero=True)
        if need > int(available * fraction):
            raise MemoryPreflightError("Estimated memory exceeds the available-memory limit.")
