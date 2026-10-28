"""Run cleaning ONCE on the REAL dataset and write the revenue report.

Diagnostic runner, not business logic: it calls the tested loader, cleaning
and revenue code. It reads the configured raw workbook and writes ONE file,
data/processed/revenue_report.txt. It never touches any other file in
data/processed. The report holds aggregate figures, stock codes and raw
product descriptions; Customer ID values are never written. No network calls.

If any reconciliation check fails, nothing is written.

Run from the project root:  python -m scripts.run_revenue_report
"""
from __future__ import annotations

import sys
import time

from src.config import PROJECT_ROOT, ConfigError, load_config
from src.ingestion.loader import DataLoadError, SchemaError, load_raw_data
from src.preprocessing.cleaning import (
    CleaningError,
    OverlapMismatchError,
    ReconciliationError,
    clean_transactions,
)
from src.reporting.revenue import (
    RevenueError,
    RevenueReconciliationError,
    build_revenue_report,
    format_revenue_report,
)

REPORT_NAME = "revenue_report.txt"
CHECKED_CODE = "TEST002"


def build_header(config, result, load_seconds: float, clean_seconds: float) -> str:
    annotated = result.annotated
    excluded = int(annotated["is_excluded"].sum())
    cleaning = config.cleaning
    listed = CHECKED_CODE in cleaning.non_product_stock_codes
    lines = [
        "=" * 78,
        "RUN INFORMATION",
        "=" * 78,
        f"Raw file: {config.data.raw_path.name}",
        f"Rows loaded and annotated: {len(annotated):,}",
        f"Retained rows: {len(annotated) - excluded:,}   Excluded rows: {excluded:,}",
        f"Sheets in order: {list(result.sheets_in_order)}",
        f"Invoices present in both sheets: {result.overlap_invoices_in_both:,}",
        f"Rows excluded as sheet_overlap_copy: {result.overlap_rows_excluded:,}",
        f"Policies: duplicate={cleaning.duplicate_policy}, "
        f"overlap={cleaning.overlap_policy}, "
        f"cancellation={cleaning.cancellation_policy}, "
        f"zero_price={cleaning.zero_price_policy}",
        f"Configured non-product stock codes: {len(cleaning.non_product_stock_codes)} "
        f"({CHECKED_CODE} listed: {'yes' if listed else 'NO'})",
        f"Load time: {load_seconds:.0f}s   Cleaning time: {clean_seconds:.0f}s",
        "Customer ID values are never written to this file.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
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
        result = clean_transactions(
            raw, config.cleaning, config.schema.required_columns
        )
    except OverlapMismatchError as exc:
        print(f"STOPPED (overlap mismatch). Nothing was removed or written.\n{exc}")
        return 1
    except (ReconciliationError, CleaningError) as exc:
        print(f"Cleaning stopped. Nothing was written.\n{type(exc).__name__}: {exc}")
        return 1
    except MemoryError:
        print("Cleaning failed: not enough memory. Close other programs and try again.")
        return 1
    clean_seconds = time.perf_counter() - started
    del raw  # the annotated table holds everything needed from here on

    print(f"Cleaned in {clean_seconds:.0f}s. Building the revenue report...")
    try:
        report = build_revenue_report(result.annotated)
    except RevenueReconciliationError as exc:
        print(
            "STOPPED: the revenue report did not reconcile. Nothing was written.\n"
            f"{exc}"
        )
        return 1
    except (RevenueError, ReconciliationError, SchemaError) as exc:
        print(f"Revenue report failed. Nothing was written.\n{type(exc).__name__}: {exc}")
        return 1

    text = (
        build_header(config, result, load_seconds, clean_seconds)
        + "\n"
        + format_revenue_report(report)
    )
    out_dir = config.data.processed_dir
    report_path = out_dir / REPORT_NAME
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write the report: {type(exc).__name__}: {exc}")
        return 1

    excluded = int(result.annotated["is_excluded"].sum())
    try:
        shown_path = report_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = report_path.name
    print(
        f"Retained {len(result.annotated) - excluded:,} rows, excluded {excluded:,}."
    )
    print(f"Net merchandise value: GBP {report.net_merchandise.value:,.2f}")
    print(f"Reconciliation checks passed: {len(report.checks)} of {len(report.checks)}")
    print(f"Report saved to {shown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())