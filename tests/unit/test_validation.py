"""Tests for src.validation.quality and src.validation.overlap.

All data here is SYNTHETIC (invented rows with hand-computed expected answers).
None of it comes from the real Online Retail II dataset.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.loader import SchemaError
from src.validation.overlap import format_overlap_report, measure_sheet_overlap
from src.validation.quality import (
    build_quality_report,
    format_report,
    unusual_stock_codes,
)

REQUIRED = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]
COLUMNS = REQUIRED + ["source_sheet"]


def _frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    return frame


# Ten invented rows. Hand-computed facts:
#  - row 2: missing Customer ID          - row 3: cancellation (C prefix)
#  - row 4: missing description, qty -5  - row 5: price 0      - row 6: price -3
#  - row 7: invoice "A1007" (letter, not C) with non-standard code POST
#  - row 8: repeats row 0 inside sheet S1
#  - row 9: sheet S2, repeats row 1 from sheet S1 only
QUALITY_ROWS = [
    ("1001", "10001", "ITEM A", 2, "2020-01-01 10:00", 1.0, 1.0, "UK", "S1"),
    ("1001", "10002", "ITEM B", 1, "2020-01-01 10:00", 2.0, 1.0, "UK", "S1"),
    ("1002", "10001", "ITEM A", 1, "2020-01-02 10:00", 1.0, None, "UK", "S1"),
    ("C1003", "10001", "ITEM A", -1, "2020-01-03 10:00", 1.0, 1.0, "UK", "S1"),
    ("1004", "10003", None, -5, "2020-01-04 10:00", 1.0, 2.0, "UK", "S1"),
    ("1005", "10004", "ITEM D", 1, "2020-01-05 10:00", 0.0, 2.0, "UK", "S1"),
    ("1006", "10005", "ITEM E", 1, "2020-01-06 10:00", -3.0, 2.0, "UK", "S1"),
    ("A1007", "POST", "POSTAGE", 1, "2020-01-07 10:00", 5.0, 2.0, "UK", "S1"),
    ("1001", "10001", "ITEM A", 2, "2020-01-01 10:00", 1.0, 1.0, "UK", "S1"),
    ("1001", "10002", "ITEM B", 1, "2020-01-01 10:00", 2.0, 1.0, "UK", "S2"),
]


@pytest.fixture()
def quality_df() -> pd.DataFrame:
    return _frame(QUALITY_ROWS)


@pytest.fixture()
def report(quality_df):
    return build_quality_report(quality_df, REQUIRED, "C")


# ------------------------------ quality report ------------------------------
def test_total_and_sheet_counts(report):
    assert report.total_rows == 10
    assert report.rows_by_sheet == {"S1": 9, "S2": 1}


def test_date_ranges(report):
    assert report.date_range_by_sheet["S1"] == (
        "2020-01-01 10:00:00",
        "2020-01-07 10:00:00",
    )
    assert report.date_range_by_sheet["S2"] == (
        "2020-01-01 10:00:00",
        "2020-01-01 10:00:00",
    )


def test_missing_values(report):
    assert report.missing_customer_id_rows == 1
    assert report.missing_by_column["Customer ID"] == 1
    assert report.missing_by_column["Description"] == 1
    assert report.missing_by_column["Price"] == 0


def test_cancellation_counts(quality_df, report):
    assert report.cancellation_rows == 1
    assert report.cancellation_rows_with_positive_quantity == 0
    changed = quality_df.copy()
    changed.loc[3, "Quantity"] = 4
    odd = build_quality_report(changed, REQUIRED, "C")
    assert odd.cancellation_rows_with_positive_quantity == 1


def test_letter_invoice_count_and_custom_prefix(quality_df, report):
    assert report.other_letter_invoice_rows == 1
    swapped = build_quality_report(quality_df, REQUIRED, "A")
    assert swapped.cancellation_rows == 1
    assert swapped.other_letter_invoice_rows == 1


def test_quantity_and_price_anomalies(report):
    assert report.non_cancellation_nonpositive_quantity_rows == 1
    assert report.zero_price_rows == 1
    assert report.negative_price_rows == 1


def test_duplicate_counts(report):
    assert report.duplicate_rows_total == 2
    assert report.duplicate_rows_within_sheet == 1
    assert report.duplicate_rows_cross_sheet_only == 1


def test_unique_counts(report):
    assert report.unique_invoices == 7
    assert report.unique_customers == 2
    assert report.unique_stock_codes == 6


def test_quality_report_does_not_modify_input(quality_df):
    before = quality_df.copy()
    build_quality_report(quality_df, REQUIRED, "C")
    unusual_stock_codes(quality_df)
    pd.testing.assert_frame_equal(before, quality_df)


def test_format_report_contains_key_lines(report):
    text = format_report(report)
    assert "Total rows: 10" in text
    assert "Missing Customer ID rows: 1 (10.00%)" in text
    assert "Cancellation rows: 1 (10.00%)" in text


def test_empty_input_report(quality_df):
    empty_report = build_quality_report(quality_df.iloc[0:0], REQUIRED)
    assert empty_report.total_rows == 0
    assert empty_report.rows_by_sheet == {}
    assert "n/a" in format_report(empty_report)


def test_quality_report_missing_column_raises(quality_df):
    with pytest.raises(SchemaError, match="Price"):
        build_quality_report(quality_df.drop(columns=["Price"]), REQUIRED)


# ------------------------------ unusual codes -------------------------------
def test_unusual_codes_counted_and_sorted():
    frame = pd.DataFrame(
        {
            "StockCode": ["85123A", "15056BL", "POST", "POST", "gift_0001_20", "DOT"],
            "Description": ["A", "B", "POSTAGE", "POSTAGE", "VOUCHER", "DOTCOM POSTAGE"],
        }
    )
    result = unusual_stock_codes(frame)
    assert result["StockCode"].tolist() == ["POST", "DOT", "gift_0001_20"]
    assert result["rows"].tolist() == [2, 1, 1]
    assert result["example_description"].tolist()[0] == "POSTAGE"


def test_all_standard_codes_returns_empty_table():
    frame = pd.DataFrame({"StockCode": ["10001", "85123A"], "Description": ["A", "B"]})
    result = unusual_stock_codes(frame)
    assert result.empty
    assert list(result.columns) == ["StockCode", "rows", "example_description"]


def test_unusual_codes_missing_column_raises():
    with pytest.raises(SchemaError, match="Description"):
        unusual_stock_codes(pd.DataFrame({"StockCode": ["POST"]}))


# --------------------------------- overlap ----------------------------------
def _row(invoice, code, quantity, date, sheet):
    return (invoice, code, "DESC", quantity, date, 1.0, 1.0, "UK", sheet)


def _identical_overlap_rows():
    return [
        _row("1", "A", 1, "2020-01-01 10:00", "S1"),
        _row("2", "A", 1, "2020-01-09 10:00", "S1"),
        _row("2", "B", 1, "2020-01-09 10:00", "S1"),
        _row("3", "A", 1, "2020-01-10 10:00", "S1"),
        _row("2", "A", 1, "2020-01-09 10:00", "S2"),
        _row("2", "B", 1, "2020-01-09 10:00", "S2"),
        _row("3", "A", 1, "2020-01-10 10:00", "S2"),
        _row("4", "A", 1, "2020-01-20 10:00", "S2"),
    ]


def test_identical_overlap_is_measured_exactly():
    result = measure_sheet_overlap(_frame(_identical_overlap_rows()), REQUIRED)
    assert result is not None
    assert (result.first_sheet, result.second_sheet) == ("S1", "S2")
    assert result.window_start == "2020-01-09 10:00:00"
    assert result.window_end == "2020-01-10 10:00:00"
    assert result.invoices_in_both == 2
    assert result.rows_in_both_first == 3
    assert result.rows_in_both_second == 3
    assert result.identical_invoices == 2
    assert result.different_invoices == 0
    assert result.example_different_invoices == ()
    assert result.window_invoices_only_in_first == 0
    assert result.window_invoices_only_in_second == 0


def test_differing_overlap_content_is_detected():
    rows = _identical_overlap_rows()
    rows[5] = _row("2", "B", 5, "2020-01-09 10:00", "S2")  # quantity differs
    result = measure_sheet_overlap(_frame(rows), REQUIRED)
    assert result.invoices_in_both == 2
    assert result.identical_invoices == 1
    assert result.different_invoices == 1
    assert result.example_different_invoices == ("2",)


def test_sheets_without_time_overlap():
    rows = [
        _row("1", "A", 1, "2020-01-01 10:00", "S1"),
        _row("2", "A", 1, "2020-01-05 10:00", "S1"),
        _row("3", "A", 1, "2020-01-10 10:00", "S2"),
        _row("4", "A", 1, "2020-01-15 10:00", "S2"),
    ]
    result = measure_sheet_overlap(_frame(rows), REQUIRED)
    assert result.invoices_in_both == 0
    assert result.window_start.startswith("none")
    assert result.identical_invoices == 0
    assert result.different_invoices == 0
    assert result.window_invoices_only_in_first == 0
    assert result.window_invoices_only_in_second == 0


def test_single_sheet_returns_none():
    rows = [_row("1", "A", 1, "2020-01-01 10:00", "S1")]
    assert measure_sheet_overlap(_frame(rows), REQUIRED) is None


def test_more_than_two_sheets_raises():
    rows = [
        _row("1", "A", 1, "2020-01-01 10:00", "S1"),
        _row("2", "A", 1, "2020-01-02 10:00", "S2"),
        _row("3", "A", 1, "2020-01-03 10:00", "S3"),
    ]
    with pytest.raises(ValueError, match="at most two"):
        measure_sheet_overlap(_frame(rows), REQUIRED)


def test_window_only_invoices_are_counted():
    rows = [
        _row("1", "A", 1, "2020-01-01 10:00", "S1"),
        _row("2", "A", 1, "2020-01-09 10:00", "S1"),
        _row("5", "A", 1, "2020-01-10 10:00", "S1"),
        _row("2", "A", 1, "2020-01-09 10:00", "S2"),
        _row("6", "A", 1, "2020-01-09 14:00", "S2"),
        _row("4", "A", 1, "2020-01-20 10:00", "S2"),
    ]
    result = measure_sheet_overlap(_frame(rows), REQUIRED)
    assert result.invoices_in_both == 1
    assert result.identical_invoices == 1
    assert result.window_invoices_only_in_first == 1
    assert result.window_invoices_only_in_second == 1


def test_overlap_does_not_modify_input():
    frame = _frame(_identical_overlap_rows())
    before = frame.copy()
    measure_sheet_overlap(frame, REQUIRED)
    pd.testing.assert_frame_equal(before, frame)


def test_overlap_missing_column_raises():
    frame = _frame(_identical_overlap_rows()).drop(columns=["Price"])
    with pytest.raises(SchemaError, match="Price"):
        measure_sheet_overlap(frame, REQUIRED)


def test_format_overlap_report_when_no_overlap_possible():
    assert "Only one sheet" in format_overlap_report(None)


def test_format_overlap_report_content():
    result = measure_sheet_overlap(_frame(_identical_overlap_rows()), REQUIRED)
    text = format_overlap_report(result)
    assert "Invoices present in BOTH sheets: 2" in text
    assert "invoices whose rows differ between sheets: 0" in text