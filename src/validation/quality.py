"""Data-quality measurements for the raw retail data.

READ-ONLY: this module counts and describes problems. It never modifies,
filters or removes rows. Decisions about what to exclude are made later, in the
cleaning step, using these measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.ingestion.loader import SOURCE_COLUMN, validate_schema

# A "standard" product code is 5 digits plus up to 2 letters (e.g. 85123A, 15056BL).
STANDARD_STOCK_CODE_PATTERN: str = r"^\d{5}[A-Za-z]{0,2}$"
NEEDED_COLUMNS: tuple[str, ...] = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
)


@dataclass(frozen=True)
class QualityReport:
    total_rows: int
    rows_by_sheet: dict[str, int]
    date_range_by_sheet: dict[str, tuple[str, str]]
    missing_by_column: dict[str, int]
    missing_customer_id_rows: int
    cancellation_rows: int
    cancellation_rows_with_positive_quantity: int
    non_cancellation_nonpositive_quantity_rows: int
    zero_price_rows: int
    negative_price_rows: int
    other_letter_invoice_rows: int
    duplicate_rows_total: int
    duplicate_rows_within_sheet: int
    duplicate_rows_cross_sheet_only: int
    unique_invoices: int
    unique_customers: int
    unique_stock_codes: int


def _bool_series(mask: pd.Series) -> pd.Series:
    return mask.fillna(False).astype(bool)


def build_quality_report(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    cancellation_prefix: str = "C",
) -> QualityReport:
    """Measure data-quality issues without changing the data."""
    validate_schema(
        list(df.columns),
        [*NEEDED_COLUMNS, *required_columns, SOURCE_COLUMN],
        "quality report input",
    )
    keys = list(required_columns)

    invoice = df["Invoice"].astype("string")
    is_cancellation = _bool_series(invoice.str.startswith(cancellation_prefix))
    starts_with_letter = _bool_series(invoice.str.match(r"^[A-Za-z]"))
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["Price"], errors="coerce")

    duplicates_total = int(df.duplicated(subset=keys).sum())
    duplicates_within = int(df.duplicated(subset=keys + [SOURCE_COLUMN]).sum())

    by_sheet = df.groupby(SOURCE_COLUMN)["InvoiceDate"].agg(["min", "max"])
    date_ranges = {
        str(sheet): (str(row["min"]), str(row["max"]))
        for sheet, row in by_sheet.iterrows()
    }
    rows_by_sheet = {
        str(sheet): int(count)
        for sheet, count in df[SOURCE_COLUMN].value_counts().sort_index().items()
    }

    return QualityReport(
        total_rows=len(df),
        rows_by_sheet=rows_by_sheet,
        date_range_by_sheet=date_ranges,
        missing_by_column={col: int(df[col].isna().sum()) for col in keys},
        missing_customer_id_rows=int(df["Customer ID"].isna().sum()),
        cancellation_rows=int(is_cancellation.sum()),
        cancellation_rows_with_positive_quantity=int(
            (is_cancellation & (quantity > 0)).sum()
        ),
        non_cancellation_nonpositive_quantity_rows=int(
            (~is_cancellation & (quantity <= 0)).sum()
        ),
        zero_price_rows=int((price == 0).sum()),
        negative_price_rows=int((price < 0).sum()),
        other_letter_invoice_rows=int((starts_with_letter & ~is_cancellation).sum()),
        duplicate_rows_total=duplicates_total,
        duplicate_rows_within_sheet=duplicates_within,
        duplicate_rows_cross_sheet_only=duplicates_total - duplicates_within,
        unique_invoices=int(df["Invoice"].nunique()),
        unique_customers=int(df["Customer ID"].nunique()),
        unique_stock_codes=int(df["StockCode"].nunique()),
    )


def unusual_stock_codes(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """List product codes that do not look like standard 5-digit codes."""
    validate_schema(list(df.columns), ["StockCode", "Description"], "stock code check")
    codes = df["StockCode"].astype("string")
    is_standard = _bool_series(codes.str.match(STANDARD_STOCK_CODE_PATTERN))
    unusual = df.loc[~is_standard, ["StockCode", "Description"]]
    if unusual.empty:
        return pd.DataFrame(columns=["StockCode", "rows", "example_description"])
    summary = (
        unusual.groupby("StockCode", dropna=False)
        .agg(rows=("Description", "size"), example_description=("Description", "first"))
        .reset_index()
        .sort_values(["rows", "StockCode"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    return summary


def format_report(report: QualityReport) -> str:
    """Return a plain-text version of the report for printing."""

    def pct(count: int) -> str:
        if report.total_rows == 0:
            return "n/a"
        return f"{100 * count / report.total_rows:.2f}%"

    lines = [
        f"Total rows: {report.total_rows}",
        f"Rows by sheet: {report.rows_by_sheet}",
        "Date range by sheet:",
    ]
    for sheet, (first, last) in report.date_range_by_sheet.items():
        lines.append(f"  {sheet}: {first} to {last}")
    lines += [
        f"Missing values by column: {report.missing_by_column}",
        f"Missing Customer ID rows: {report.missing_customer_id_rows} "
        f"({pct(report.missing_customer_id_rows)})",
        f"Cancellation rows: {report.cancellation_rows} "
        f"({pct(report.cancellation_rows)})",
        "  ...of which Quantity > 0 (unexpected): "
        f"{report.cancellation_rows_with_positive_quantity}",
        "Non-cancellation rows with Quantity <= 0: "
        f"{report.non_cancellation_nonpositive_quantity_rows}",
        f"Price == 0 rows: {report.zero_price_rows}",
        f"Price < 0 rows: {report.negative_price_rows}",
        "Non-cancellation invoices starting with a letter: "
        f"{report.other_letter_invoice_rows}",
        f"Duplicate rows (repeat of an earlier row): {report.duplicate_rows_total}",
        f"  ...within the same sheet: {report.duplicate_rows_within_sheet}",
        f"  ...repeating a row from the other sheet only: "
        f"{report.duplicate_rows_cross_sheet_only}",
        f"Unique invoices: {report.unique_invoices}",
        f"Unique customers (known IDs): {report.unique_customers}",
        f"Unique stock codes: {report.unique_stock_codes}",
    ]
    return "\n".join(lines)