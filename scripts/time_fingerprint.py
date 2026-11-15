"""Time the logical DataFrame fingerprint on the REAL dataset (Phase 1, Step 1).

Read-only: loads the raw workbook, runs the existing cleaning code, then
fingerprints the annotated DataFrame. Writes NO files. Prints counts, timings,
column kinds and the OUTER digest only. It never prints cell values and never
prints per-column digests (those include the Customer ID column).

By default it fingerprints three times: twice with the default chunk size
(determinism and timing) and once with a different chunk size (chunk
independence). --quick runs a single fingerprint.

Memory: the fingerprint processes one chunk at a time. If the Step 0 sampler
(scripts/inspect_schema.py) can be imported, the sampled peak increase is
printed; it is a lower bound (short spikes can be missed).

Run from the project root:  python -m scripts.time_fingerprint [--quick]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import pandas as pd

from src.config import ConfigError, load_config
from src.ingestion.loader import DataLoadError, load_raw_data
from src.persistence.hashing import (
    DEFAULT_CHUNK_ROWS,
    FINGERPRINT_VERSION,
    FingerprintError,
    differing_columns,
    fingerprint_dataframe,
)
from src.preprocessing.cleaning import (
    CleaningError,
    OverlapMismatchError,
    clean_transactions,
)

try:  # optional: its absence only disables the memory figure
    from scripts.inspect_schema import PeakSampler
except Exception:  # noqa: BLE001
    PeakSampler = None

ALTERNATE_CHUNK_ROWS = 33_333


def run_once(df: pd.DataFrame, chunk_rows: int):
    seconds: dict[str, float] = {}
    sampler = PeakSampler() if PeakSampler is not None else None
    started = time.perf_counter()
    if sampler is not None:
        with sampler:
            result = fingerprint_dataframe(
                df, chunk_rows=chunk_rows, column_seconds=seconds
            )
    else:
        result = fingerprint_dataframe(df, chunk_rows=chunk_rows, column_seconds=seconds)
    elapsed = time.perf_counter() - started
    increase = getattr(sampler, "increase_mb", None) if sampler is not None else None
    return result, seconds, elapsed, increase


def describe(label: str, chunk_rows: int, elapsed: float, increase) -> str:
    memory = (
        f"sampled process memory +{increase:,.0f} MB (lower bound)"
        if increase is not None
        else "memory not measured"
    )
    return f"{label} (chunk {chunk_rows:,} rows): {elapsed:,.1f} s, {memory}"


def report_fingerprints(df: pd.DataFrame, quick: bool) -> int:
    """Fingerprint `df`, print timings and checks. Returns a process exit code."""
    print(f"Fingerprinting {len(df):,} rows x {df.shape[1]} columns...")
    print(f"Fingerprint format version: {FINGERPRINT_VERSION}")

    try:
        first, seconds, elapsed, increase = run_once(df, DEFAULT_CHUNK_ROWS)
    except FingerprintError as exc:
        print(f"FINGERPRINT FAILED: {exc}")
        return 1
    except MemoryError:
        print("Fingerprint failed: not enough memory.")
        return 1
    print(describe("Run 1", DEFAULT_CHUNK_ROWS, elapsed, increase))
    print(f"Outer digest: {first.digest}")

    print()
    print(f"{'column':<26}{'kind':<18}{'seconds':>9}")
    for name, kind in first.columns:
        print(f"{name:<26}{kind:<18}{seconds[name]:>9.2f}")

    repeat_ok = chunk_ok = None
    if not quick:
        try:
            second, _, elapsed, increase = run_once(df, DEFAULT_CHUNK_ROWS)
            third, _, elapsed_alt, increase_alt = run_once(df, ALTERNATE_CHUNK_ROWS)
        except FingerprintError as exc:
            print(f"FINGERPRINT FAILED: {exc}")
            return 1
        except MemoryError:
            print("Fingerprint failed: not enough memory.")
            return 1
        print()
        print(describe("Run 2", DEFAULT_CHUNK_ROWS, elapsed, increase))
        print(describe("Run 3", ALTERNATE_CHUNK_ROWS, elapsed_alt, increase_alt))
        repeat_ok = second == first
        chunk_ok = third == first
        if not chunk_ok:
            print(
                "Columns that differ with the other chunk size: "
                f"{differing_columns(first, third)}"
            )

    print()
    print(f"Rows fingerprinted: {first.rows:,}")
    print(
        "Repeat run gives the same fingerprint: "
        f"{'not checked (--quick)' if repeat_ok is None else repeat_ok}"
    )
    print(
        "Different chunk size gives the same fingerprint: "
        f"{'not checked (--quick)' if chunk_ok is None else chunk_ok}"
    )
    return 0 if repeat_ok in (None, True) and chunk_ok in (None, True) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time the logical DataFrame fingerprint on the real dataset."
    )
    parser.add_argument("--quick", action="store_true", help="run a single fingerprint")
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration problem: {exc}")
        return 1

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

    print(f"Loaded {len(raw):,} rows in {load_seconds:.0f}s. Cleaning...")
    started = time.perf_counter()
    try:
        result = clean_transactions(raw, config.cleaning, config.schema.required_columns)
    except OverlapMismatchError as exc:
        print(f"STOPPED (overlap mismatch).\n{exc}")
        return 1
    except CleaningError as exc:
        print(f"Cleaning stopped: {type(exc).__name__}: {exc}")
        return 1
    except MemoryError:
        print("Cleaning failed: not enough memory. Close other programs and try again.")
        return 1
    clean_seconds = time.perf_counter() - started
    del raw
    gc.collect()

    print(f"Cleaned in {clean_seconds:.0f}s.")
    return report_fingerprints(result.annotated, args.quick)


if __name__ == "__main__":
    sys.exit(main())