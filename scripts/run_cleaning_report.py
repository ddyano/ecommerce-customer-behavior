"""Run the cleaning step on the REAL dataset and write a review report.

Diagnostic runner, not business logic: it calls the tested loader and cleaning
code, then adds a few extra summaries. Reads the configured raw file and
writes ONE text file (data/processed/cleaning_report.txt) with aggregate
counts and sample raw rows. Customer ID values are never written.
Makes no network calls.

Run from the project root:  python -m scripts.run_cleaning_report
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from src.config import ConfigError, load_config
from src.ingestion.loader import DataLoadError, load_raw_data
from src.preprocessing import rules
from src.preprocessing.cleaning import (
    CleaningError,
    CleaningResult,
    OverlapMismatchError,
    audit_uncovered_unusual_codes,
    format_reason_summary,
)

REPORT_NAME = "cleaning_report.txt"
SAMPLE_COLUMNS = [
    "row_id", "source_sheet", "Invoice", "StockCode", "Description",
    "Quantity", "Price", "has_customer_id", "reason_code",
]
SAMPLE_REASONS = [
    rules.UNRECOGNIZED_INVOICE_PREFIX,
    rules.CANCELLATION_NONNEGATIVE_QUANTITY,
    rules.NEGATIVE_PRICE,
    rules.INVALID_FIELD,
]
BREAKDOWN_REASONS = [rules.NONPOSITIVE_QUANTITY, rules.ZERO_PRICE]
MAX_SAMPLE_ROWS = 15
TOP_N = 10
AUDIT_ROWS = 40


def _table(frame: pd.DataFrame) -> str:
    return "(none)" if frame.empty else frame.to_string(index=False)


def _flag(series: pd.Series) -> "pd.Series":
    return series.fillna(False).to_numpy(dtype=bool)


def _top_values(series: pd.Series, top_n: int) -> pd.DataFrame:
    counts = series.astype("string").fillna("<missing>").value_counts().head(top_n)
    return counts.rename_axis("value").reset_index(name="rows")


def build_report(
    annotated: pd.DataFrame,
    result: CleaningResult,
    config,
    load_seconds: float,
    clean_seconds: float,
) -> str:
    lines: list[str] = []

    def section(title: str) -> None:
        lines.extend(["", "=" * 78, title, "=" * 78])

    reason = annotated["reason_code"]
    section("1. RUN INFORMATION")
    lines += [
        f"Rows loaded and annotated: {len(annotated)}",
        f"Load time: {load_seconds:.0f}s   Cleaning time: {clean_seconds:.0f}s",
        f"Annotated table memory (not peak): "
        f"{annotated.memory_usage(deep=True).sum() / 1e6:,.0f} MB",
        f"Sheets in order: {list(result.sheets_in_order)}",
        f"Invoices present in both sheets: {result.overlap_invoices_in_both}",
        f"Rows excluded as sheet_overlap_copy: {result.overlap_rows_excluded}",
        "Reconciliation: passed (clean_transactions raises an error if it fails)",
    ]

    section("2. ROWS, UNITS AND LINE VALUE BY REASON (first match wins)")
    lines.append(format_reason_summary(result.reason_summary))

    section("3. INVOICE-PREFIX CHECK (letter at the start of Invoice)")
    invoice = annotated["Invoice"].astype("string")
    first_char = invoice.str.extract(r"^([A-Za-z])")[0]
    has_letter = _flag(first_char.notna())
    lines.append("Rows by leading letter (case-sensitive):")
    lines.append(first_char.value_counts().rename_axis("letter").reset_index(name="rows").pipe(_table))
    by_prefix = (
        annotated.loc[has_letter, ["reason_code"]]
        .assign(prefix=first_char[has_letter])
        .groupby(["prefix", "reason_code"])
        .size()
        .reset_index(name="rows")
    )
    lines += ["", "Reasons by leading letter:", _table(by_prefix)]
    lower_c = _flag(invoice.str.startswith("c"))
    lines += [
        "",
        f"Invoices starting with lowercase 'c': {int(lower_c.sum())} rows",
        _table(reason[lower_c].value_counts().rename_axis("reason").reset_index(name="rows")),
    ]

    section("4. LISTED CODES (final reason is a listed-code reason)")
    listed_reasons = [rules.LISTED_NON_PRODUCT_CODE, rules.LISTED_UNIDENTIFIABLE_CODE]
    listed = annotated.loc[reason.isin(listed_reasons).to_numpy(dtype=bool)]
    by_code = (
        listed.groupby(["reason_code", "StockCode"]).size().reset_index(name="rows")
        .sort_values(["reason_code", "rows"], ascending=[True, False])
    )
    lines.append(_table(by_code))
    configured = set(config.cleaning.non_product_stock_codes) | set(
        config.cleaning.unidentifiable_stock_codes
    )
    unmatched = sorted(configured - set(listed["StockCode"].astype(str)))
    lines.append(
        "Configured codes with no row under a listed-code reason "
        f"(may be caught by an earlier rule or absent): {unmatched}"
    )

    section("5. VOUCHER AND MERCHANDISE ROW COUNTS BY CUSTOMER-ID PRESENCE")
    lines.append(
        _table(
            annotated.groupby(["analytic_role", "has_customer_id"])
            .size().reset_index(name="rows")
        )
    )

    section("6. EXACT-DUPLICATE FLAG (informational; rows are kept unless policy is 'drop')")
    flagged = annotated["is_exact_duplicate"].to_numpy(dtype=bool)
    lines.append(f"Rows flagged as exact duplicates: {int(flagged.sum())}")
    lines.append(
        _table(reason[flagged].value_counts().rename_axis("final_reason").reset_index(name="rows"))
    )

    section("7. UNUSUAL STOCK CODES NOT COVERED BY ANY LIST (audit trail; not excluded)")
    audit = audit_uncovered_unusual_codes(annotated, config.cleaning)
    lines.append(f"Distinct uncovered unusual codes: {len(audit)} (showing up to {AUDIT_ROWS})")
    lines.append(_table(audit.head(AUDIT_ROWS)))

    section("8. WHAT IS IN THE LARGE EXCLUDED GROUPS (most common raw values)")
    for code in BREAKDOWN_REASONS:
        subset = annotated.loc[(reason == code).to_numpy(dtype=bool)]
        lines += [
            "",
            f"--- {code}: {len(subset)} rows; "
            f"{int(subset['Description'].isna().sum())} with missing Description; "
            f"{int((~subset['has_customer_id']).sum())} without Customer ID ---",
            "Most common Description values:",
            _table(_top_values(subset["Description"], TOP_N)),
            "Most common StockCode values:",
            _table(_top_values(subset["StockCode"], TOP_N)),
        ]

    section("9. SAMPLE RAW ROWS FOR SMALL EXCLUDED GROUPS (Customer ID not shown)")
    for code in SAMPLE_REASONS:
        subset = annotated.loc[(reason == code).to_numpy(dtype=bool), SAMPLE_COLUMNS]
        lines += ["", f"--- {code}: {len(subset)} rows (showing up to {MAX_SAMPLE_ROWS}) ---"]
        lines.append(_table(subset.head(MAX_SAMPLE_ROWS)))

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
    load_seconds = time.perf_counter() - started

    print(f"Loaded {len(raw)} rows in {load_seconds:.0f}s. Cleaning...")
    started = time.perf_counter()
    try:
        result = clean_transactions_safely(raw, config)
    except OverlapMismatchError as exc:
        print(f"STOPPED (overlap mismatch). Nothing was removed or written.\n{exc}")
        return 1
    except CleaningError as exc:
        print(f"Cleaning stopped: {exc}")
        return 1
    clean_seconds = time.perf_counter() - started

    report = build_report(result.annotated, result, config, load_seconds, clean_seconds)
    out_dir = config.data.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_NAME
    report_path.write_text(report, encoding="utf-8")

    excluded = int(result.annotated["is_excluded"].sum())
    print(
        f"Cleaned in {clean_seconds:.0f}s: {len(result.annotated) - excluded} retained, "
        f"{excluded} excluded."
    )
    print(f"Report saved to {report_path.relative_to(config.data.processed_dir.parent.parent)}")
    return 0


def clean_transactions_safely(raw: pd.DataFrame, config) -> CleaningResult:
    from src.preprocessing.cleaning import clean_transactions

    return clean_transactions(raw, config.cleaning, config.schema.required_columns)


if __name__ == "__main__":
    sys.exit(main())