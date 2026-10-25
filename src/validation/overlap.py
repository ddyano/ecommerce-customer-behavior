"""Measure how two workbook sheets overlap in time and in content.

READ-ONLY: counts and compares rows; never modifies, filters or removes data.
The result informs the cleaning rule for the overlapping period.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.ingestion.loader import SOURCE_COLUMN, validate_schema

MAX_EXAMPLES: int = 5


@dataclass(frozen=True)
class OverlapReport:
    first_sheet: str
    second_sheet: str
    window_start: str
    window_end: str
    invoices_in_both: int
    rows_in_both_first: int
    rows_in_both_second: int
    identical_invoices: int
    different_invoices: int
    example_different_invoices: tuple[str, ...]
    window_invoices_only_in_first: int
    window_invoices_only_in_second: int


def measure_sheet_overlap(
    df: pd.DataFrame, required_columns: Sequence[str]
) -> OverlapReport | None:
    """Compare two sheets; return None if there is only one source."""
    validate_schema(
        list(df.columns),
        ["Invoice", "InvoiceDate", *required_columns, SOURCE_COLUMN],
        "overlap input",
    )
    starts = df.groupby(SOURCE_COLUMN)["InvoiceDate"].min().sort_values()
    sheets = [str(name) for name in starts.index]
    if len(sheets) < 2:
        return None
    if len(sheets) > 2:
        raise ValueError(
            f"Overlap check supports at most two sheets, found {len(sheets)}."
        )

    first, second = sheets
    keys = list(required_columns)
    is_first = df[SOURCE_COLUMN] == first
    is_second = df[SOURCE_COLUMN] == second

    shared = set(df.loc[is_first, "Invoice"]) & set(df.loc[is_second, "Invoice"])
    in_shared = df["Invoice"].isin(shared)

    different: set[str] = set()
    if shared:
        subset = (
            df.loc[in_shared, keys + [SOURCE_COLUMN]].astype("string").fillna("<NA>")
        )
        counts = (
            subset.groupby(keys + [SOURCE_COLUMN])
            .size()
            .unstack(SOURCE_COLUMN, fill_value=0)
            .reindex(columns=[first, second], fill_value=0)
        )
        mismatch = counts[first] != counts[second]
        different = {str(v) for v in counts.index[mismatch].get_level_values("Invoice")}

    start = df.loc[is_second, "InvoiceDate"].min()
    end = df.loc[is_first, "InvoiceDate"].max()
    only_first = only_second = 0
    if start <= end:
        in_window = df["InvoiceDate"].between(start, end)
        window_first = set(df.loc[in_window & is_first, "Invoice"])
        window_second = set(df.loc[in_window & is_second, "Invoice"])
        only_first = len(window_first - window_second)
        only_second = len(window_second - window_first)
        window_start, window_end = str(start), str(end)
    else:
        window_start = window_end = "none (sheets do not overlap in time)"

    return OverlapReport(
        first_sheet=first,
        second_sheet=second,
        window_start=window_start,
        window_end=window_end,
        invoices_in_both=len(shared),
        rows_in_both_first=int((in_shared & is_first).sum()),
        rows_in_both_second=int((in_shared & is_second).sum()),
        identical_invoices=len(shared) - len(different),
        different_invoices=len(different),
        example_different_invoices=tuple(sorted(different)[:MAX_EXAMPLES]),
        window_invoices_only_in_first=only_first,
        window_invoices_only_in_second=only_second,
    )


def format_overlap_report(report: OverlapReport | None) -> str:
    """Return a plain-text version of the overlap report."""
    if report is None:
        return "Only one sheet/source found: no overlap to measure."
    return "\n".join(
        [
            f"Sheets compared: '{report.first_sheet}' (earlier start) and "
            f"'{report.second_sheet}'",
            f"Overlap window: {report.window_start} to {report.window_end}",
            f"Invoices present in BOTH sheets: {report.invoices_in_both}",
            f"  rows for those invoices in first sheet: {report.rows_in_both_first}",
            f"  rows for those invoices in second sheet: {report.rows_in_both_second}",
            f"  invoices with identical rows in both sheets: {report.identical_invoices}",
            f"  invoices whose rows differ between sheets: {report.different_invoices}",
            f"  examples of differing invoices: {list(report.example_different_invoices)}",
            "Invoices inside the window found only in first sheet: "
            f"{report.window_invoices_only_in_first}",
            "Invoices inside the window found only in second sheet: "
            f"{report.window_invoices_only_in_second}",
        ]
    )