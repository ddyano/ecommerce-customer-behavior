"""Step 0 of Phase 1: inspect the REAL annotated DataFrame before designing the
Parquet schema. Nothing is persisted.

It reads the raw workbook with the existing loader, runs the existing cleaning
code once, then reports dtypes, nulls, memory, categorical candidates, columns
that duplicate each other, and the result of IN-MEMORY Parquet round trips (no
Parquet file is written).

Memory design:
  * Sections 1-8 (always run) make NO full-column conversion to category and NO
    full-column unique()/nunique()/value_counts(). Distinct counting is
    BOUNDED: it scans in SCAN_CHUNK_ROWS-row chunks, keeps a running set, and
    gives up once the set exceeds a cap (reported as "> cap"). Each column is
    counted ONCE and the result is reused by every later section.
  * High-cardinality columns never keep their distinct values; their category
    memory is APPROXIMATE (or unavailable above the cap).
  * Other column scans (Arrow type inference, type census, Customer ID checks,
    numeric comparisons) run in chunks. The one remaining whole-frame temporary
    in sections 1-8 is summarize_reasons() in section 1 (existing cleaning
    code), plus whatever pandas/NumPy/PyArrow allocate internally.
  * Column comparisons never convert a whole column to Python objects.
  * Before each round trip a pre-flight check estimates the extra memory phase
    by phase. The estimate uses unmeasured heuristics and is NOT a guarantee.
  * The report prints the measured (sampled) peak increase next to the estimate
    so the heuristics can be calibrated.

Options:
  --skip-roundtrips   skip both round-trip sections
  --force-roundtrips  run them even if the pre-flight check says skip

Its only output file is data/processed/schema_inspection.txt. Customer ID
values are never printed or written. No network calls.

Run from the project root:  python -m scripts.inspect_schema
"""
from __future__ import annotations

import argparse
import functools
import gc
import io
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, ConfigError, load_config
from src.ingestion.loader import SOURCE_COLUMN, DataLoadError, load_raw_data
from src.preprocessing import rules
from src.preprocessing.cleaning import (
    CleaningError,
    OverlapMismatchError,
    clean_transactions,
    summarize_reasons,
)

REPORT_NAME = "schema_inspection.txt"
MB = 1024 * 1024
CATEGORICAL_CANDIDATES = (
    "reason_code",
    "disposition",
    "analytic_role",
    "Country",
    "source_sheet",
)
HIGH_CARDINALITY = ("Invoice", "StockCode", "Description")
DUPLICATE_INFO_COLUMNS = (
    "quantity_num",
    "price_num",
    "invoice_datetime",
    "has_customer_id",
)

# Bounded distinct counting. A column whose distinct count exceeds its cap is
# reported as "> cap" and gets no category-memory estimate.
CARDINALITY_CAP = 200_000
CANDIDATE_CARDINALITY_CAP = 10_000  # the five candidates are expected to be tiny
SCAN_CHUNK_ROWS = 100_000  # rows per chunk for every always-run column scan

# pandas stores category codes in the smallest signed integer that fits. These
# limits mirror pandas' rule from memory (UNVERIFIED); section 10 compares the
# estimate with an actual conversion for the five candidate columns.
INT8_LIMIT = 2**7 - 1
INT16_LIMIT = 2**15 - 1
INT32_LIMIT = 2**31 - 1

BASIS_MEASURED = "measured distinct values"
BASIS_APPROXIMATE = "approximate: column memory x distinct/rows"
BASIS_UNAVAILABLE = "unavailable: distinct count above cap"

# ---------------------------------------------------------------------------
# Memory heuristics. ALL of these are UNMEASURED upper-bound guesses, used only
# for the pre-flight check. They are NOT guarantees of safety either way.
# F = the annotated frame's pandas memory (deep), measured at run time.
# ---------------------------------------------------------------------------
ARROW_TABLE_FACTOR = 1.0  # pa.Table.from_pandas copy, as a multiple of F
PARQUET_BUFFER_FACTOR = 0.3  # in-memory compressed Parquet bytes, as a multiple of F
DECODED_TABLE_FACTOR = 1.0  # pq.read_table result, as a multiple of F
RESTORED_FRAME_FACTOR = 1.0  # .to_pandas() result, as a multiple of F
CATEGORICAL_BUILD_BYTES_PER_ROW = 100  # transient while converting ONE column to category
CATEGORICAL_CODES_BYTES_PER_ROW = 10  # the five code arrays that stay alive afterwards
COMPARE_CHUNK_ROWS = 100_000  # rows per chunk in the object-comparison path
COMPARE_TEMP_BYTES_PER_ROW = 1024  # generous bound on chunk temporaries per row
FIXED_ALLOWANCE_MB = 256.0  # interpreter, allocator and library slack
WARN_FRACTION_OF_TOTAL_RAM = 0.75  # warn if projected in-use memory exceeds this
SKIP_FRACTION_OF_AVAILABLE_RAM = 0.80  # skip if estimated need exceeds this share of free RAM
SAMPLE_INTERVAL_SECONDS = 0.05


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def mb(value: float) -> str:
    return f"{float(value) / MB:,.1f}"


def _short(text: object, limit: int = 90) -> str:
    cleaned = str(text).replace("\n", " ")
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def _table(frame: pd.DataFrame) -> str:
    return "(none)" if frame.empty else frame.to_string(index=False)


# --------------------------------------------------------------------------
# Memory measurement (Windows system calls; every failure returns None)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _win_api():
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    return ctypes, Counters, MemoryStatus, kernel32, psapi


def process_memory_mb() -> dict[str, float] | None:
    """Memory of THIS process in MB (Windows only); None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        ctypes, Counters, _, kernel32, psapi = _win_api()
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
        return {
            "current_working_set": counters.WorkingSetSize / MB,
            "peak_working_set": counters.PeakWorkingSetSize / MB,
            "peak_commit": counters.PeakPagefileUsage / MB,
        }
    except Exception:  # noqa: BLE001 - best-effort measurement only
        return None


def system_memory_mb() -> dict[str, float] | None:
    """Whole-computer memory in MB (Windows only); None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        ctypes, _, MemoryStatus, kernel32, _ = _win_api()
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return {
            "total_physical": status.ullTotalPhys / MB,
            "available_physical": status.ullAvailPhys / MB,
            "available_commit": status.ullAvailPageFile / MB,
            "memory_load_percent": float(status.dwMemoryLoad),
        }
    except Exception:  # noqa: BLE001 - best-effort measurement only
        return None


def arrow_pool_mb() -> dict[str, object] | None:
    """PyArrow's own memory pool, where available."""
    try:
        import pyarrow as pa

        pool = pa.default_memory_pool()
        peak = pool.max_memory()
        return {
            "backend": pool.backend_name,
            "allocated": pool.bytes_allocated() / MB,
            "peak": None if peak is None or peak < 0 else peak / MB,
        }
    except Exception:  # noqa: BLE001 - best-effort measurement only
        return None


def format_memory(memory: dict[str, float] | None) -> str:
    if memory is None:
        return "not available on this platform"
    return (
        f"current {memory['current_working_set']:,.0f} MB, "
        f"peak working set {memory['peak_working_set']:,.0f} MB, "
        f"peak commit {memory['peak_commit']:,.0f} MB"
    )


def format_system(memory: dict[str, float] | None) -> str:
    if memory is None:
        return "not available on this platform"
    return (
        f"total RAM {memory['total_physical']:,.0f} MB, "
        f"available {memory['available_physical']:,.0f} MB, "
        f"in use {memory['memory_load_percent']:.0f}%, "
        f"available commit {memory['available_commit']:,.0f} MB"
    )


def format_arrow(info: dict[str, object] | None) -> str:
    if info is None:
        return "not available"
    peak = info["peak"]
    peak_text = "not reported" if peak is None else f"{peak:,.0f} MB"
    return (
        f"backend {info['backend']}, allocated now {info['allocated']:,.0f} MB, "
        f"peak so far {peak_text}"
    )


class PeakSampler:
    """Samples this process's working set in a background thread (Windows only).

    The reported increase is a LOWER BOUND: short spikes between samples are missed.
    """

    def __init__(self, interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self.interval = interval
        self.available = False
        self.baseline = 0.0
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            sample = process_memory_mb()
            if sample is not None:
                self.peak = max(self.peak, sample["current_working_set"])

    def __enter__(self) -> "PeakSampler":
        sample = process_memory_mb()
        self.available = sample is not None
        if sample is not None:
            self.baseline = self.peak = sample["current_working_set"]
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        sample = process_memory_mb()
        if sample is not None:
            self.peak = max(self.peak, sample["current_working_set"])
        return False

    @property
    def increase_mb(self) -> float | None:
        if not self.available:
            return None
        return max(0.0, self.peak - self.baseline)


def _measured(sampler: PeakSampler) -> str:
    increase = sampler.increase_mb
    if increase is None:
        return "not available on this platform"
    return (
        f"+{increase:,.0f} MB (sampled every {int(SAMPLE_INTERVAL_SECONDS * 1000)} ms; "
        "short spikes can be missed)"
    )


# --------------------------------------------------------------------------
# Bounded distinct counting (replaces unique()/nunique()/value_counts())
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Distinct:
    """Result of one bounded scan of a column.

    count    exact distinct count when not exceeded; otherwise a LOWER BOUND (> cap)
    values   the distinct values, kept ONLY when requested and not exceeded
    """

    count: int
    exceeded: bool
    cap: int
    values: tuple | None = None


def bounded_distinct(
    series: pd.Series,
    cap: int,
    keep_values: bool = False,
    chunk_rows: int = SCAN_CHUNK_ROWS,
) -> Distinct:
    """Count distinct non-missing values without a full-column unique().

    Scans in chunks and keeps a running set. The set can reach at most about
    cap + chunk_rows entries before the scan stops, so memory is bounded by the
    cap and not by the number of rows. Past the cap the scan stops and reports
    exceeded=True (the count is then only a lower bound).
    """
    seen: set = set()
    for start in range(0, len(series), chunk_rows):
        chunk = series.iloc[start : start + chunk_rows].dropna()
        if chunk.empty:
            continue
        seen.update(list(chunk.unique()))
        if len(seen) > cap:
            return Distinct(len(seen), True, cap, None)
    return Distinct(len(seen), False, cap, tuple(seen) if keep_values else None)


def cap_for(column: str) -> int:
    return CANDIDATE_CARDINALITY_CAP if column in CATEGORICAL_CANDIDATES else CARDINALITY_CAP


def fmt_distinct(distinct: Distinct) -> str:
    return f"> {distinct.cap:,}" if distinct.exceeded else f"{distinct.count:,}"


def _code_itemsize(n_categories: int) -> int:
    """Bytes per category code: pandas uses the smallest signed integer that fits."""
    if n_categories < INT8_LIMIT:
        return 1
    if n_categories < INT16_LIMIT:
        return 2
    if n_categories < INT32_LIMIT:
        return 4
    return 8


def estimate_category_memory(series: pd.Series, distinct: Distinct) -> dict[str, object]:
    """Estimate the memory of `series` as a category WITHOUT converting it.

    * Distinct values kept (candidate columns): categories memory is measured on
      those values (at most CANDIDATE_CARDINALITY_CAP of them).
    * Distinct values not kept (high-cardinality columns): categories memory is
      APPROXIMATED as the column's own memory x distinct / rows. This assumes the
      distinct values are the same average size as the rows.
    * Distinct count above the cap: no estimate (unavailable).
    Integer codes use the smallest signed width that fits (rule from memory,
    UNVERIFIED). The categories index's lazily built lookup table is ignored.
    """
    rows = len(series)
    current = float(series.memory_usage(deep=True, index=False))
    if distinct.exceeded:
        return {
            "basis": BASIS_UNAVAILABLE,
            "rows": rows,
            "unique": None,
            "current_mb": current / MB,
            "codes_mb": None,
            "categories_mb": None,
            "estimated_mb": None,
            "saving_mb": None,
        }
    n_categories = distinct.count
    codes = float(rows * _code_itemsize(n_categories))
    if distinct.values is not None:
        categories = float(
            pd.Series(list(distinct.values), dtype=series.dtype).memory_usage(
                deep=True, index=False
            )
        )
        basis = BASIS_MEASURED
    else:
        categories = current * n_categories / rows if rows else 0.0
        basis = BASIS_APPROXIMATE
    estimated = codes + categories
    return {
        "basis": basis,
        "rows": rows,
        "unique": n_categories,
        "current_mb": current / MB,
        "codes_mb": codes / MB,
        "categories_mb": categories / MB,
        "estimated_mb": estimated / MB,
        "saving_mb": (current - estimated) / MB,
    }


# --------------------------------------------------------------------------
# Pre-flight estimate and decision
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Estimate:
    frame_mb: float
    rows: int
    phases: tuple[tuple[str, float], ...]
    peak_phase_mb: float
    compare_temp_mb: float
    build_mb: float
    extra_mb: float


def estimate_roundtrip(frame_mb: float, rows: int, categorical: bool) -> Estimate:
    """Phase-by-phase extra-memory estimate (heuristic, unmeasured).

    Phases run one after another, so the estimate is the LARGEST phase plus a
    fixed allowance, not the sum.
    """
    codes_mb = rows * CATEGORICAL_CODES_BYTES_PER_ROW / MB if categorical else 0.0
    build_mb = (
        rows * CATEGORICAL_BUILD_BYTES_PER_ROW / MB + codes_mb if categorical else 0.0
    )
    compare_mb = COMPARE_CHUNK_ROWS * COMPARE_TEMP_BYTES_PER_ROW / MB
    phases: list[tuple[str, float]] = []
    if categorical:
        phases.append(("convert five columns to category, one column at a time", build_mb))
    phases.append(
        (
            "write: Arrow copy + Parquet buffer",
            (ARROW_TABLE_FACTOR + PARQUET_BUFFER_FACTOR) * frame_mb + codes_mb,
        )
    )
    phases.append(
        (
            "read: Parquet buffer + decoded Arrow table + restored frame",
            (PARQUET_BUFFER_FACTOR + DECODED_TABLE_FACTOR + RESTORED_FRAME_FACTOR)
            * frame_mb
            + codes_mb,
        )
    )
    phases.append(
        (
            "compare: restored frame + chunk temporaries",
            RESTORED_FRAME_FACTOR * frame_mb + compare_mb + codes_mb,
        )
    )
    peak = max(value for _, value in phases)
    return Estimate(
        frame_mb=frame_mb,
        rows=rows,
        phases=tuple(phases),
        peak_phase_mb=peak,
        compare_temp_mb=compare_mb,
        build_mb=build_mb,
        extra_mb=peak + FIXED_ALLOWANCE_MB,
    )


@dataclass(frozen=True)
class Preflight:
    run: bool
    lines: list[str]


def assess_roundtrip(
    label: str,
    estimate: Estimate,
    mode: str,
    system: dict[str, float] | None,
) -> Preflight:
    """Decide whether to run an in-memory round trip. Heuristic, not a guarantee."""
    need = estimate.extra_mb
    lines = [
        f"Pre-flight for {label}: frame {estimate.frame_mb:,.0f} MB, {estimate.rows:,} rows.",
        "Estimated extra memory by phase (heuristic, unmeasured; phases run one after another):",
    ]
    lines += [f"  - {name}: {value:,.0f} MB" for name, value in estimate.phases]
    lines.append(
        f"Largest phase {estimate.peak_phase_mb:,.0f} MB + fixed allowance "
        f"{FIXED_ALLOWANCE_MB:,.0f} MB = estimated extra {need:,.0f} MB."
    )
    if mode == "skip":
        lines.append("SKIPPED: --skip-roundtrips was given.")
        return Preflight(False, lines)
    if system is None:
        if mode == "force":
            lines.append(
                "WARNING: system memory could not be read; running because "
                "--force-roundtrips was given."
            )
            return Preflight(True, lines)
        lines.append(
            "SKIPPED: system memory could not be read. "
            "Use --force-roundtrips to run anyway."
        )
        return Preflight(False, lines)

    total = system["total_physical"]
    available = system["available_physical"]
    in_use = total - available
    projected = in_use + need
    lines.append(
        f"System RAM: total {total:,.0f} MB, available {available:,.0f} MB, "
        f"in use {in_use:,.0f} MB."
    )
    lines.append(
        f"Projected in use during the round trip: {projected:,.0f} MB "
        f"({100 * projected / total:.0f}% of total RAM)."
    )
    if projected > WARN_FRACTION_OF_TOTAL_RAM * total:
        lines.append(
            f"WARNING: projected use exceeds {WARN_FRACTION_OF_TOTAL_RAM:.0%} of total "
            "RAM. This is an assessment threshold, not a guarantee either way."
        )
    if need > SKIP_FRACTION_OF_AVAILABLE_RAM * available:
        if mode == "force":
            lines.append(
                f"WARNING: estimated need exceeds {SKIP_FRACTION_OF_AVAILABLE_RAM:.0%} "
                "of available RAM; running because --force-roundtrips was given. "
                "This can raise a MemoryError or slow the computer."
            )
            return Preflight(True, lines)
        lines.append(
            f"SKIPPED: estimated extra memory exceeds "
            f"{SKIP_FRACTION_OF_AVAILABLE_RAM:.0%} of currently available RAM. "
            "Close other programs and rerun, or use --force-roundtrips."
        )
        return Preflight(False, lines)
    lines.append(
        "Proceeding: the estimate fits within the available-RAM limit. Passing this "
        "check does not guarantee success."
    )
    return Preflight(True, lines)


def _announce(decision: Preflight) -> None:
    for line in decision.lines:
        if line.startswith(("SKIPPED", "WARNING", "Proceeding")):
            print(line)


# --------------------------------------------------------------------------
# Memory-bounded column comparison (never converts a whole column to objects)
# --------------------------------------------------------------------------
def _compare_path(left: pd.Series, right: pd.Series) -> str:
    """Pick the cheapest correct comparison: 'codes', 'numpy' or 'objects'."""
    left_dtype, right_dtype = left.dtype, right.dtype
    if (
        isinstance(left_dtype, pd.CategoricalDtype)
        and isinstance(right_dtype, pd.CategoricalDtype)
        and left_dtype.ordered == right_dtype.ordered
        and left_dtype.categories.equals(right_dtype.categories)
    ):
        return "codes"
    if isinstance(left_dtype, np.dtype) and isinstance(right_dtype, np.dtype):
        if left_dtype.kind in "biuf" and right_dtype.kind in "biuf":
            return "numpy"
        if left_dtype.kind == right_dtype.kind and left_dtype.kind in "Mm":
            return "numpy"
    return "objects"


def _chunk_equal(left: pd.Series, right: pd.Series, path: str) -> bool:
    if path == "codes":
        return bool(
            np.array_equal(left.cat.codes.to_numpy(), right.cat.codes.to_numpy())
        )
    missing = left.isna().to_numpy()
    if not np.array_equal(missing, right.isna().to_numpy()):
        return False
    keep = ~missing
    if not keep.any():
        return True
    if path == "numpy":
        first, second = left.to_numpy(), right.to_numpy()
        if not keep.all():
            first, second = first[keep], second[keep]
        return bool(np.array_equal(first, second))
    first = left.astype(object).to_numpy()
    second = right.astype(object).to_numpy()
    if not keep.all():
        first, second = first[keep], second[keep]
    return bool((first == second).all())


def _same_values(first: pd.Series, second: pd.Series) -> bool:
    """True if both series hold the same values, missing values in the same places.

    Temporaries are bounded by COMPARE_CHUNK_ROWS, not by the length of the column.
    """
    if len(first) != len(second):
        return False
    path = _compare_path(first, second)
    for start in range(0, len(first), COMPARE_CHUNK_ROWS):
        stop = start + COMPARE_CHUNK_ROWS
        if not _chunk_equal(first.iloc[start:stop], second.iloc[start:stop], path):
            return False
    return True


def _compare_frames(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in before.columns:
        if column not in after.columns:
            rows.append(
                {
                    "column": column,
                    "dtype_before": str(before[column].dtype),
                    "dtype_after": "MISSING",
                    "dtype_changed": True,
                    "nulls_before": int(before[column].isna().sum()),
                    "nulls_after": -1,
                    "values_equal": False,
                }
            )
            continue
        left, right = before[column], after[column]
        rows.append(
            {
                "column": column,
                "dtype_before": str(left.dtype),
                "dtype_after": str(right.dtype),
                "dtype_changed": str(left.dtype) != str(right.dtype),
                "nulls_before": int(left.isna().sum()),
                "nulls_after": int(right.isna().sum()),
                "values_equal": _same_values(left, right),
            }
        )
    return pd.DataFrame(rows)


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed: list[str] = []
        self.roundtrips: dict[str, str] = {}
        # Estimated category MB per column, only where measured (section 4).
        self.estimates: dict[str, float] = {}
        # One bounded scan per column (section 4), reused by later sections.
        self.distinct: dict[str, Distinct] = {}

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def section(self, title: str, func: Callable[[], None]) -> None:
        self.lines.extend(["", "=" * 78, title, "=" * 78])
        try:
            func()
        except Exception as exc:  # noqa: BLE001 - isolate each section (MemoryError included)
            self.failed.append(title)
            self.lines.append(f"SECTION FAILED: {type(exc).__name__}: {_short(exc, 200)}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def candidate_values(report: Report, df: pd.DataFrame, column: str) -> tuple[str, ...]:
    """Sorted distinct values of a low-cardinality candidate column.

    Reuses the section 4 scan when it is available (no second scan); otherwise
    runs ONE bounded scan. Raises if the column is not actually low-cardinality.
    """
    distinct = report.distinct.get(column)
    if distinct is None or (distinct.values is None and not distinct.exceeded):
        distinct = bounded_distinct(
            df[column], CANDIDATE_CARDINALITY_CAP, keep_values=True
        )
        report.distinct[column] = distinct
    if distinct.exceeded:
        raise ValueError(
            f"{column} has more than {distinct.cap:,} distinct values, so it is not "
            "a low-cardinality column"
        )
    return tuple(sorted(str(value) for value in distinct.values))


def snapshot(df: pd.DataFrame) -> dict[str, object]:
    """Cheap fingerprint: shape, dtypes and null counts ONLY (not cell values)."""
    return {
        "shape": df.shape,
        "dtypes": {column: str(df[column].dtype) for column in df.columns},
        "nulls": {column: int(df[column].isna().sum()) for column in df.columns},
    }


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def sec_run_info(report: Report, ctx: dict) -> None:
    df, result = ctx["df"], ctx["result"]
    try:
        import pyarrow

        arrow_version = pyarrow.__version__
    except Exception:  # noqa: BLE001
        arrow_version = "not installed"
    excluded = int(df["is_excluded"].sum())
    report.add(
        f"Python {sys.version.split()[0]}; pandas {pd.__version__}; "
        f"numpy {np.__version__}; pyarrow {arrow_version}",
        f"Rows: {len(df):,}   Columns: {df.shape[1]}   Index: {type(df.index).__name__}",
        f"Load time: {ctx['load_seconds']:.0f}s   Cleaning time: {ctx['clean_seconds']:.0f}s",
        f"Annotated frame memory (deep): {ctx['frame_mb']:,.1f} MB",
        f"Retained: {len(df) - excluded:,}   Excluded: {excluded:,}",
        f"Round-trip mode: {ctx['mode']}",
        f"System memory at start: {format_system(ctx['system_start'])}",
        "Bounded distinct counting: cap "
        f"{CARDINALITY_CAP:,} distinct values per column ({CANDIDATE_CARDINALITY_CAP:,} "
        f"for the five candidate columns), scanned in chunks of {SCAN_CHUNK_ROWS:,} rows; "
        "above the cap a column is reported as '> cap' and gets no category estimate.",
        "Memory heuristics (pre-flight only, unmeasured): "
        f"Arrow copy {ARROW_TABLE_FACTOR}x, Parquet buffer {PARQUET_BUFFER_FACTOR}x, "
        f"decoded table {DECODED_TABLE_FACTOR}x, restored frame {RESTORED_FRAME_FACTOR}x "
        f"of frame memory; categorical build {CATEGORICAL_BUILD_BYTES_PER_ROW} B/row; "
        f"comparison chunks of {COMPARE_CHUNK_ROWS:,} rows at "
        f"{COMPARE_TEMP_BYTES_PER_ROW} B/row; fixed allowance "
        f"{FIXED_ALLOWANCE_MB:,.0f} MB; warn above {WARN_FRACTION_OF_TOTAL_RAM:.0%} of "
        f"total RAM; skip above {SKIP_FRACTION_OF_AVAILABLE_RAM:.0%} of available RAM.",
        "CleaningResult values that will go in the manifest:",
        f"  sheets_in_order = {list(result.sheets_in_order)}",
        f"  overlap_invoices_in_both = {result.overlap_invoices_in_both}",
        f"  overlap_rows_excluded = {result.overlap_rows_excluded}",
    )
    same = summarize_reasons(df).equals(result.reason_summary)
    report.add(
        "reason_summary recomputed from the annotated frame equals the original: "
        f"{same}"
    )
    for stage, memory in ctx["memory"].items():
        report.add(f"Process memory {stage}: {format_memory(memory)}")


def _arrow_type_chunked(series: pd.Series) -> str:
    """Arrow type inferred in chunks, so temporaries are bounded by one chunk."""
    import pyarrow as pa

    if len(series) == 0:
        return "n/a (no rows)"
    types: set[str] = set()
    for start in range(0, len(series), SCAN_CHUNK_ROWS):
        chunk = series.iloc[start : start + SCAN_CHUNK_ROWS]
        try:
            table = pa.Table.from_pandas(chunk.to_frame(), preserve_index=False)
            types.add(str(table.schema.field(0).type))
        except Exception as exc:  # noqa: BLE001
            types.add(f"ERROR {type(exc).__name__}: {_short(exc, 50)}")
    if len(types) > 1:
        types.discard("null")  # an all-null chunk is not a conflicting type
    if len(types) == 1:
        return next(iter(types))
    return "MIXED ACROSS CHUNKS: " + "; ".join(sorted(types))


def sec_schema(report: Report, df: pd.DataFrame) -> None:
    report.add(
        f"Arrow types are inferred in chunks of {SCAN_CHUNK_ROWS:,} rows, one column at a "
        "time, so temporaries are bounded by one chunk. If chunks disagree the type is "
        "reported as MIXED ACROSS CHUNKS (all-null chunks are ignored). The type Arrow "
        "would choose for the full column in a later write could differ; section 9 shows "
        "the actual outcome if it runs.",
        "",
    )
    rows = []
    for position, column in enumerate(df.columns, start=1):
        series = df[column]
        nulls = int(series.isna().sum())
        rows.append(
            {
                "pos": position,
                "column": column,
                "pandas_dtype": str(series.dtype),
                "arrow_type": _arrow_type_chunked(series),
                "nulls": nulls,
                "null_pct": round(100 * nulls / len(df), 3) if len(df) else 0.0,
            }
        )
    report.add(_table(pd.DataFrame(rows)))


def sec_text_census(report: Report, df: pd.DataFrame) -> None:
    for column in df.columns:
        series = df[column]
        if series.dtype == object:
            kinds: Counter = Counter()
            for start in range(0, len(series), SCAN_CHUNK_ROWS):
                chunk = series.iloc[start : start + SCAN_CHUNK_ROWS]
                kinds.update(type(value) for value in chunk)
            listing = ", ".join(
                f"{kind.__name__}={count:,}" for kind, count in kinds.most_common()
            )
            report.add(f"{column}: object dtype; Python types found: {listing}")
        elif pd.api.types.is_string_dtype(series.dtype):
            report.add(f"{column}: {series.dtype} (text dtype)")


def sec_cardinality_and_memory(report: Report, df: pd.DataFrame) -> None:
    per_column = df.memory_usage(deep=True, index=False)
    rows = []
    for column in df.columns:
        try:
            distinct = bounded_distinct(
                df[column],
                cap_for(column),
                keep_values=column in CATEGORICAL_CANDIDATES,
            )
            report.distinct[column] = distinct
            shown = fmt_distinct(distinct)
        except Exception as exc:  # noqa: BLE001 - one column must not hide the others
            shown = f"ERROR {type(exc).__name__}: {_short(exc, 40)}"
        rows.append(
            {
                "column": column,
                "distinct": shown,
                "memory_mb": round(float(per_column[column]) / MB, 1),
            }
        )
    report.add(
        "Distinct counts come from ONE bounded scan per column (reused by later "
        f"sections); '> N' means the cap was exceeded and the true count is unknown.",
        _table(pd.DataFrame(rows)),
        f"Total: {mb(per_column.sum())} MB",
    )

    report.add(
        "",
        "ESTIMATED memory if converted to category. NO conversion is performed in "
        "this section.",
        f"  '{BASIS_MEASURED}': the candidate column's own distinct values (at most "
        f"{CANDIDATE_CARDINALITY_CAP:,}) were measured.",
        f"  '{BASIS_APPROXIMATE}': high-cardinality columns never keep their distinct "
        "values; categories memory is scaled from the column's own memory, which "
        "assumes the distinct values are the same average size as the rows. Rough.",
        f"  '{BASIS_UNAVAILABLE}': not estimated, to avoid a large allocation.",
        "Integer codes use the smallest signed width that fits (rule from memory, "
        "UNVERIFIED). Section 10 compares the candidate estimates with an actual "
        "conversion if it runs; the high-cardinality columns are never converted, so "
        "their estimates stay unverified.",
    )
    estimates = []
    for group, columns in (
        ("candidate", CATEGORICAL_CANDIDATES),
        ("high-cardinality (never converted)", HIGH_CARDINALITY),
    ):
        for column in columns:
            if column not in df.columns:
                continue
            series = df[column]
            distinct = report.distinct.get(column)
            base = {
                "group": group,
                "column": column,
                "dtype": str(series.dtype),
                "rows": f"{len(series):,}",
            }
            if distinct is None:
                estimates.append({**base, "distinct": "unknown", "basis": "unavailable: count failed"})
                continue
            try:
                result = estimate_category_memory(series, distinct)
            except Exception as exc:  # noqa: BLE001
                estimates.append(
                    {
                        **base,
                        "distinct": fmt_distinct(distinct),
                        "basis": f"ERROR {type(exc).__name__}: {_short(exc, 40)}",
                    }
                )
                continue
            if result["basis"] == BASIS_MEASURED:
                report.estimates[column] = float(result["estimated_mb"])

            def rounded(key: str) -> float | None:
                value = result[key]
                return None if value is None else round(float(value), 2)

            estimates.append(
                {
                    **base,
                    "distinct": fmt_distinct(distinct),
                    "distinct_per_row": (
                        None
                        if distinct.exceeded or not len(series)
                        else round(distinct.count / len(series), 4)
                    ),
                    "basis": result["basis"],
                    "current_mb": rounded("current_mb"),
                    "est_codes_mb": rounded("codes_mb"),
                    "est_categories_mb": rounded("categories_mb"),
                    "est_category_mb": rounded("estimated_mb"),
                    "est_saving_mb": rounded("saving_mb"),
                }
            )
    report.add(_table(pd.DataFrame(estimates)))


def sec_classification_values(report: Report, df: pd.DataFrame) -> None:
    expected = {
        "reason_code": list(rules.REASON_CODES),
        "disposition": [rules.RETAINED, rules.EXCLUDED],
        "analytic_role": list(rules.ROLES),
    }
    report.add("(Distinct values are reused from the section 4 scan; no second scan.)")
    for column, allowed in expected.items():
        observed = list(candidate_values(report, df, column))
        unexpected = [value for value in observed if value not in allowed]
        unused = [value for value in allowed if value not in observed]
        report.add(
            f"{column}: {len(observed)} observed of {len(allowed)} defined; "
            f"unexpected values: {unexpected}; defined but unused: {unused}"
        )
    for column in ("Country", SOURCE_COLUMN):
        values = list(candidate_values(report, df, column))
        report.add(f"{column}: {len(values)} distinct; values: {values}")


def sec_customer_id(report: Report, df: pd.DataFrame) -> None:
    series = df["Customer ID"]
    unreadable = 0
    whole = True
    for start in range(0, len(series), SCAN_CHUNK_ROWS):
        chunk = series.iloc[start : start + SCAN_CHUNK_ROWS]
        numeric = pd.to_numeric(chunk, errors="coerce")
        unreadable += int((numeric.isna() & chunk.notna()).sum())
        present = numeric.dropna().to_numpy(dtype=float)
        if len(present) and not bool(np.all(present == np.floor(present))):
            whole = False
    distinct = report.distinct.get("Customer ID")
    if distinct is None:
        distinct = bounded_distinct(series, CARDINALITY_CAP)
        report.distinct["Customer ID"] = distinct
    report.add(
        f"dtype: {series.dtype}",
        f"missing: {int(series.isna().sum()):,}",
        f"distinct known IDs: {fmt_distinct(distinct)} (reused from the section 4 scan)",
        f"known IDs not readable as numbers: {unreadable:,}",
        f"all known IDs are whole numbers: {whole}",
        "(ID values are not printed.)",
    )


def sec_dates(report: Report, df: pd.DataFrame) -> None:
    for column in ("InvoiceDate", "invoice_datetime"):
        try:
            series = df[column]
            report.add(
                f"{column}: dtype {series.dtype}; unit {getattr(series.dtype, 'unit', None)}; "
                f"timezone {getattr(series.dtype, 'tz', None)}; "
                f"earliest {series.min()}; latest {series.max()}"
            )
        except Exception as exc:  # noqa: BLE001
            report.add(f"{column}: could not summarise ({type(exc).__name__}: {_short(exc)})")


def _numeric_chunks_equal(raw: pd.Series, parsed: pd.Series) -> bool:
    """Compare a raw column (coerced to numbers) with its parsed twin, in chunks."""
    for start in range(0, len(raw), SCAN_CHUNK_ROWS):
        first = pd.to_numeric(
            raw.iloc[start : start + SCAN_CHUNK_ROWS], errors="coerce"
        ).to_numpy(dtype=float)
        second = parsed.iloc[start : start + SCAN_CHUNK_ROWS].to_numpy(dtype=float)
        if not np.array_equal(first, second, equal_nan=True):
            return False
    return True


def sec_duplicate_information(report: Report, df: pd.DataFrame) -> None:
    for raw_name, parsed_name in (("Quantity", "quantity_num"), ("Price", "price_num")):
        try:
            same = _numeric_chunks_equal(df[raw_name], df[parsed_name])
            report.add(f"{raw_name} vs {parsed_name}: identical values = {same}")
        except Exception as exc:  # noqa: BLE001
            report.add(
                f"{raw_name} vs {parsed_name}: could not compare "
                f"({type(exc).__name__}: {_short(exc)})"
            )
    try:
        same = _same_values(df["InvoiceDate"], df["invoice_datetime"])
        report.add(f"InvoiceDate vs invoice_datetime: identical values = {same}")
    except Exception as exc:  # noqa: BLE001
        report.add(
            f"InvoiceDate vs invoice_datetime: could not compare "
            f"({type(exc).__name__}: {_short(exc)})"
        )
    try:
        flag_same = bool(
            np.array_equal(
                df["has_customer_id"].to_numpy(dtype=bool),
                df["Customer ID"].notna().to_numpy(dtype=bool),
            )
        )
        report.add(f"has_customer_id vs Customer ID notna: identical = {flag_same}")
    except Exception as exc:  # noqa: BLE001
        report.add(
            f"has_customer_id vs Customer ID notna: could not compare "
            f"({type(exc).__name__}: {_short(exc)})"
        )
    total = 0.0
    for column in DUPLICATE_INFO_COLUMNS:
        value = float(df[column].memory_usage(deep=True, index=False))
        total += value
        report.add(f"memory of {column}: {mb(value)} MB")
    report.add(f"memory of these four columns together: {mb(total)} MB")


def _parquet_roundtrip(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    size = buffer.tell()
    del table
    buffer.seek(0)
    restored = pq.read_table(buffer).to_pandas()
    return restored, size


def sec_roundtrip_as_is(report: Report, df: pd.DataFrame, frame_mb: float, mode: str) -> None:
    estimate = estimate_roundtrip(frame_mb, len(df), categorical=False)
    decision = assess_roundtrip(
        "the as-is round trip", estimate, mode, system_memory_mb()
    )
    report.add(*decision.lines)
    _announce(decision)
    report.roundtrips["as_is"] = "ran" if decision.run else "skipped"
    if not decision.run:
        return
    before = process_memory_mb()
    with PeakSampler() as transfer:
        restored, size = _parquet_roundtrip(df)
    with PeakSampler() as comparing:
        comparison = _compare_frames(df, restored)
    report.add(
        f"In-memory Parquet size (snappy): {mb(size)} MB",
        f"Rows before/after: {len(df):,} / {len(restored):,}",
        f"Column order identical: {list(df.columns) == list(restored.columns)}",
        _table(comparison),
        f"Columns whose dtype changed: {int(comparison['dtype_changed'].sum())}",
        f"Columns whose values differ: {int((~comparison['values_equal']).sum())}",
        f"Estimated largest write/read phase: {estimate.peak_phase_mb:,.0f} MB; "
        f"measured peak increase during write + read: {_measured(transfer)}",
        f"Estimated comparison temporaries (bound): {estimate.compare_temp_mb:,.0f} MB; "
        f"measured peak increase during comparison: {_measured(comparing)}",
    )
    del restored
    gc.collect()
    report.add(
        f"Process memory before: {format_memory(before)}",
        f"Process memory after: {format_memory(process_memory_mb())}",
        f"Arrow memory pool after: {format_arrow(arrow_pool_mb())}",
    )


def _category_definitions(df: pd.DataFrame, report: Report) -> dict[str, list[str]]:
    return {
        "reason_code": list(rules.REASON_CODES),
        "disposition": [rules.RETAINED, rules.EXCLUDED],
        "analytic_role": list(rules.ROLES),
        "Country": list(candidate_values(report, df, "Country")),
        SOURCE_COLUMN: list(candidate_values(report, df, SOURCE_COLUMN)),
    }


def sec_roundtrip_optimized(report: Report, df: pd.DataFrame, frame_mb: float, mode: str) -> None:
    estimate = estimate_roundtrip(frame_mb, len(df), categorical=True)
    decision = assess_roundtrip(
        "the categorical round trip", estimate, mode, system_memory_mb()
    )
    report.add(*decision.lines)
    _announce(decision)
    report.roundtrips["categorical"] = "ran" if decision.run else "skipped"
    if not decision.run:
        return
    before = process_memory_mb()
    definitions = _category_definitions(df, report)
    nulls_before = {column: int(df[column].isna().sum()) for column in definitions}

    # Shallow copy: all other columns are shared with df (copy-on-write), and the
    # five categorical columns are built one at a time. df itself is only read.
    with PeakSampler() as building:
        optimized = df.copy(deep=False)
        for column, categories in definitions.items():
            optimized[column] = df[column].astype(
                pd.CategoricalDtype(categories=categories)
            )
    outside = {}
    for column in definitions:
        extra_missing = int(optimized[column].isna().sum()) - nulls_before[column]
        if extra_missing:
            outside[column] = extra_missing
    after_mb = float(optimized.memory_usage(deep=True).sum()) / MB
    report.add(
        f"Values outside the category definitions: {outside if outside else 'none'}",
        f"Frame memory as is: {frame_mb:,.1f} MB; with the five columns as category: "
        f"{after_mb:,.1f} MB; saving {frame_mb - after_mb:,.1f} MB",
        f"Estimated categorical-build transient: {estimate.build_mb:,.0f} MB; measured "
        f"peak increase while building the categorical frame: {_measured(building)}",
    )

    per_column = []
    for column in definitions:
        current = float(df[column].memory_usage(deep=True, index=False)) / MB
        actual = float(optimized[column].memory_usage(deep=True, index=False)) / MB
        estimated = report.estimates.get(column)
        per_column.append(
            {
                "column": column,
                "current_mb": round(current, 2),
                "actual_category_mb": round(actual, 2),
                "actual_saving_mb": round(current - actual, 2),
                "estimated_category_mb": None if estimated is None else round(estimated, 2),
                "estimate_error_mb": None if estimated is None else round(estimated - actual, 2),
            }
        )
    report.add(
        "",
        "Actual per-column memory against the section 4 estimate (small differences "
        "are expected for reason_code, disposition and analytic_role, because this "
        "conversion uses the full defined category lists, including unused values):",
        _table(pd.DataFrame(per_column)),
    )

    with PeakSampler() as transfer:
        restored, size = _parquet_roundtrip(optimized)
    detail_rows = []
    with PeakSampler() as comparing:
        for column, categories in definitions.items():
            series = restored[column]
            is_category = isinstance(series.dtype, pd.CategoricalDtype)
            detail_rows.append(
                {
                    "column": column,
                    "dtype_after_read": str(series.dtype),
                    "restored_as_category": is_category,
                    "categories_identical": bool(
                        is_category and list(series.cat.categories) == categories
                    ),
                    "values_equal_to_original": _same_values(df[column], series),
                }
            )
        comparison = _compare_frames(optimized, restored)
    report.add(
        "",
        f"In-memory Parquet size (snappy, categoricals): {mb(size)} MB",
        _table(pd.DataFrame(detail_rows)),
        "Full comparison against the optimized frame:",
        f"Columns whose dtype changed: {int(comparison['dtype_changed'].sum())}",
        f"Columns whose values differ: {int((~comparison['values_equal']).sum())}",
        f"Estimated largest write/read phase: {estimate.peak_phase_mb:,.0f} MB; "
        f"measured peak increase during write + read: {_measured(transfer)}",
        f"Estimated comparison temporaries (bound): {estimate.compare_temp_mb:,.0f} MB; "
        f"measured peak increase during comparisons: {_measured(comparing)}",
    )
    del restored, optimized
    gc.collect()
    report.add(
        f"Process memory before: {format_memory(before)}",
        f"Process memory after: {format_memory(process_memory_mb())}",
        f"Arrow memory pool after: {format_arrow(arrow_pool_mb())}",
    )


def sec_unchanged(report: Report, df: pd.DataFrame, before: dict[str, object]) -> None:
    after = snapshot(df)
    same = before == after
    report.add(
        "Scope: this check compares shape, dtypes and null counts ONLY. It does NOT "
        "prove that every original cell value is unchanged. (Nothing in this script "
        "assigns into the original frame; all column assignments go to a shallow copy.)",
        "Original frame shape, dtypes and null counts unchanged by this inspection: "
        f"{same}",
    )
    if not same:
        for key in ("shape", "dtypes", "nulls"):
            if before[key] != after[key]:
                report.add(f"DIFFERENCE in {key}: before {before[key]} / after {after[key]}")


REMAINING_OPERATIONS = (
    "Remaining memory-intensive operations in sections 1-8 (all figures are UNMEASURED estimates):",
    "  - Section 1: summarize_reasons() (existing cleaning code, unchanged) builds a few "
    "full-length temporary arrays; roughly 50 MB at worst. This is the one whole-frame "
    "temporary left in sections 1-8.",
    f"  - Section 2: Arrow type inference, one column at a time in {SCAN_CHUNK_ROWS:,}-row "
    "chunks (probably under 10 MB), plus a 1 MB null mask per column.",
    "  - Section 3: type counting in chunks; negligible.",
    f"  - Section 4: bounded distinct counting; a set of at most about "
    f"{CARDINALITY_CAP + SCAN_CHUNK_ROWS:,} entries per column ({CANDIDATE_CARDINALITY_CAP + SCAN_CHUNK_ROWS:,} "
    "for the five candidates), freed after each column; tens of MB at worst. "
    "memory_usage(deep=True) walks Python objects (CPU time, no large allocation).",
    "  - Section 5: none (reuses section 4).",
    "  - Sections 6 and 8: chunked numeric checks; a few MB per chunk, plus two 1 MB "
    "boolean arrays.",
    "  - Section 7: min()/max() on two date columns; possibly a 1 MB mask.",
    "  - Not controlled: temporaries that pandas, NumPy and PyArrow allocate internally.",
)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def check_arrow_dependency() -> tuple[str | None, list[str]]:
    """Check that PyArrow can be used. Returns (version, problems).

    Step 0 REQUIRES PyArrow in every mode, including --skip-roundtrips: section 2
    infers Arrow types, and sections 9-10 write and read Parquet in memory. An
    empty problems list means pyarrow and pyarrow.parquet both imported.
    """
    try:
        import pyarrow

        version = str(pyarrow.__version__)
    except Exception as exc:  # noqa: BLE001 - report any import failure clearly
        return None, [
            f"pyarrow cannot be imported ({type(exc).__name__}: {_short(exc, 120)})"
        ]
    try:
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return version, [
            f"pyarrow {version} is installed but pyarrow.parquet cannot be imported "
            f"({type(exc).__name__}: {_short(exc, 120)})"
        ]
    return version, []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the real annotated DataFrame before designing the Parquet schema."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-roundtrips",
        action="store_true",
        help="skip both in-memory Parquet round-trip sections",
    )
    group.add_argument(
        "--force-roundtrips",
        action="store_true",
        help="run the round trips even if the pre-flight memory check says skip",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "skip" if args.skip_roundtrips else "force" if args.force_roundtrips else "auto"
    arrow_version, arrow_problems = check_arrow_dependency()
    if arrow_problems:
        print(
            "STOPPED before loading any data: Step 0 requires PyArrow, "
            "even with --skip-roundtrips."
        )
        for problem in arrow_problems:
            print(f"  - {problem}")
        print("Fix (downloads from PyPI; changes only this virtual environment):")
        print('  python -m pip install --only-binary=:all: "pyarrow==26.0.0"')
        print("Then rerun. Step 0 does not change requirements.txt; that happens in Step 1.")
        return 1
    print(f"PyArrow {arrow_version}: available")

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration problem: {exc}")
        return 1

    system_start = system_memory_mb()
    print(f"Round-trip mode: {mode}")
    print(f"System memory at start: {format_system(system_start)}")
    memory: dict[str, dict[str, float] | None] = {"at start": process_memory_mb()}
    print(f"Loading {config.data.raw_path.name} (usually a minute or two)...")
    started = time.perf_counter()
    try:
        raw = load_raw_data(
            config.data.raw_path, config.schema.required_columns, config.data.sheet_names
        )
    except DataLoadError as exc:
        print(f"Load failed: {exc}")
        return 1
    except MemoryError:
        print("Load failed: not enough memory. Close other programs and try again.")
        return 1
    load_seconds = time.perf_counter() - started
    memory["after load"] = process_memory_mb()

    print(f"Loaded {len(raw):,} rows in {load_seconds:.0f}s. Cleaning...")
    started = time.perf_counter()
    try:
        result = clean_transactions(raw, config.cleaning, config.schema.required_columns)
    except OverlapMismatchError as exc:
        print(f"STOPPED (overlap mismatch). Nothing was written.\n{exc}")
        return 1
    except CleaningError as exc:
        print(f"Cleaning stopped: {type(exc).__name__}: {exc}")
        return 1
    except MemoryError:
        print("Cleaning failed: not enough memory. Close other programs and try again.")
        return 1
    clean_seconds = time.perf_counter() - started
    memory["after cleaning"] = process_memory_mb()

    del raw
    gc.collect()
    memory["after releasing the raw frame"] = process_memory_mb()

    df = result.annotated
    frame_mb = float(df.memory_usage(deep=True).sum()) / MB
    before_snapshot = snapshot(df)
    ctx = {
        "df": df,
        "result": result,
        "load_seconds": load_seconds,
        "clean_seconds": clean_seconds,
        "memory": memory,
        "mode": mode,
        "system_start": system_start,
        "frame_mb": frame_mb,
    }
    print("Inspecting (this takes a few minutes)...")
    report = Report()
    report.add("SCHEMA INSPECTION (Phase 1, Step 0)")
    with PeakSampler() as always_run:
        report.section("1. RUN INFORMATION", lambda: sec_run_info(report, ctx))
        report.section("2. COLUMNS: PANDAS DTYPE, ARROW TYPE, NULLS", lambda: sec_schema(report, df))
        report.section("3. TEXT COLUMNS: PYTHON TYPES FOUND", lambda: sec_text_census(report, df))
        report.section("4. DISTINCT COUNTS (BOUNDED), MEMORY AND ESTIMATED CATEGORY MEMORY", lambda: sec_cardinality_and_memory(report, df))
        report.section("5. CLASSIFICATION AND LOW-CARDINALITY VALUES", lambda: sec_classification_values(report, df))
        report.section("6. CUSTOMER ID (TYPE ONLY, NO VALUES)", lambda: sec_customer_id(report, df))
        report.section("7. DATE COLUMNS", lambda: sec_dates(report, df))
        report.section("8. DERIVED COLUMNS THAT MAY DUPLICATE OTHERS", lambda: sec_duplicate_information(report, df))
    report.section("9. IN-MEMORY PARQUET ROUND TRIP, FRAME AS IT IS", lambda: sec_roundtrip_as_is(report, df, frame_mb, mode))
    report.section("10. IN-MEMORY PARQUET ROUND TRIP, CATEGORICALS", lambda: sec_roundtrip_optimized(report, df, frame_mb, mode))
    final_memory = process_memory_mb()
    report.section(
        "11. MEMORY AT THE END (peak includes any round trips that ran)",
        lambda: report.add(
            format_memory(final_memory),
            f"System memory now: {format_system(system_memory_mb())}",
            f"Arrow memory pool: {format_arrow(arrow_pool_mb())}",
            "Sections 1-8 (always run; no round trips, no category conversions): "
            f"measured peak increase {_measured(always_run)}",
            *REMAINING_OPERATIONS,
        ),
    )
    report.section(
        "12. READ-ONLY CHECK (SHAPE, DTYPES, NULL COUNTS ONLY)",
        lambda: sec_unchanged(report, df, before_snapshot),
    )

    out_dir = config.data.processed_dir
    report_path = out_dir / REPORT_NAME
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.text(), encoding="utf-8")
    except OSError as exc:
        print(f"Could not write the report: {type(exc).__name__}: {exc}")
        return 1

    try:
        shown_path = report_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = report_path.name
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Load {load_seconds:.0f}s, cleaning {clean_seconds:.0f}s")
    print(f"Round trips: {report.roundtrips if report.roundtrips else 'not reached'}")
    print(f"Memory at end: {format_memory(final_memory)}")
    print(f"Sections that failed: {report.failed if report.failed else 'none'}")
    print(f"Report saved to {shown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())