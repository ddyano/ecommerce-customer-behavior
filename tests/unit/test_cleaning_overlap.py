"""Tests for overlap handling, duplicates, policy switches and the code audit.

All data here is SYNTHETIC (invented rows). None of it comes from the real
Online Retail II dataset.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.config import CleaningConfig
from src.ingestion.loader import SchemaError
from src.preprocessing.cleaning import (
    CleaningError,
    OverlapMismatchError,
    audit_uncovered_unusual_codes,
    clean_transactions,
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
WINDOW = "2020-01-09 10:00"


# ------------------------------- test helpers --------------------------------
def make_row(
    invoice,
    code,
    quantity,
    price,
    *,
    sheet="S1",
    date="2020-01-01 10:00",
    customer=1.0,
    description="SYNTHETIC ITEM",
):
    return (invoice, code, description, quantity, date, price, customer, "UK", sheet)


def make_frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    return frame


def make_config(**overrides) -> CleaningConfig:
    base = CleaningConfig(
        cancellation_invoice_prefix="C",
        non_product_stock_codes=("POST", "M", "B"),
        duplicate_policy="flag",
        overlap_policy="keep_earlier_sheet",
        cancellation_policy="return",
        zero_price_policy="exclude",
        unidentifiable_stock_codes=("DCGSSGIRL",),
        voucher_stock_code_prefixes=("gift_0001_",),
    )
    return replace(base, **overrides)


def run(rows, **overrides):
    return clean_transactions(make_frame(rows), make_config(**overrides), REQUIRED)


def reasons(rows, **overrides) -> list[str]:
    return run(rows, **overrides).annotated["reason_code"].tolist()


def s1_rows():
    """Sheet S1: invoice 1 (before the window) and invoice 2 (two lines)."""
    return [
        make_row("1", "85123A", 2, 1.0, date="2020-01-01 10:00"),
        make_row("2", "85123A", 1, 1.0, date=WINDOW),
        make_row("2", "85123B", 1, 2.0, date=WINDOW),
    ]


def s2(code, quantity, price, **kwargs):
    """A row of invoice 2 in sheet S2."""
    kwargs.setdefault("date", WINDOW)
    return make_row("2", code, quantity, price, sheet="S2", **kwargs)


def identical_overlap_rows():
    """S1 invoices 1 and 2; S2 repeats invoice 2 exactly and adds invoice 3."""
    return s1_rows() + [
        s2("85123A", 1, 1.0),
        s2("85123B", 1, 2.0),
        make_row("3", "85123A", 1, 1.0, sheet="S2", date="2020-01-12 10:00"),
    ]


# ============================ overlap: normal behaviour =======================
def test_identical_overlap_later_copy_is_excluded():
    result = run(identical_overlap_rows())
    annotated = result.annotated
    assert annotated["reason_code"].tolist() == [
        "merchandise_sale",
        "merchandise_sale",
        "merchandise_sale",
        "sheet_overlap_copy",
        "sheet_overlap_copy",
        "merchandise_sale",
    ]
    assert result.sheets_in_order == ("S1", "S2")
    assert result.overlap_invoices_in_both == 1
    assert result.overlap_rows_excluded == 2
    assert result.overlap_rows_excluded == int(
        (annotated["reason_code"] == "sheet_overlap_copy").sum()
    )
    # Overlap copies are not also counted as exact duplicates.
    assert annotated["is_exact_duplicate"].tolist() == [False] * 6


def test_sheet_order_follows_dates_not_row_order():
    rows = [
        make_row("2", "85123A", 1, 1.0, sheet="S2", date=WINDOW),
        make_row("1", "85123A", 1, 1.0, sheet="S1", date="2020-01-01 10:00"),
        make_row("2", "85123A", 1, 1.0, sheet="S1", date=WINDOW),
    ]
    result = run(rows)
    assert result.sheets_in_order == ("S1", "S2")
    assert result.annotated["reason_code"].tolist() == [
        "sheet_overlap_copy",
        "merchandise_sale",
        "merchandise_sale",
    ]


@pytest.mark.parametrize(
    "appearance,expected_sheets",
    [
        pytest.param(["S1", "S2"], ("S1", "S2"), id="S1_appears_first"),
        pytest.param(["S2", "S1"], ("S2", "S1"), id="S2_appears_first"),
    ],
)
def test_tied_sheet_dates_use_order_of_appearance(appearance, expected_sheets):
    rows = [
        make_row("C2", "85123A", -1, 1.0, date=WINDOW, sheet=name)
        for name in appearance
    ]
    result = run(rows)
    assert result.sheets_in_order == expected_sheets
    assert result.annotated["reason_code"].tolist() == [
        "merchandise_return",
        "sheet_overlap_copy",
    ]


def test_single_sheet_has_no_overlap():
    result = run([make_row("1", "85123A", 1, 1.0), make_row("2", "85123A", 1, 1.0)])
    assert result.sheets_in_order == ("S1",)
    assert result.overlap_invoices_in_both == 0
    assert result.overlap_rows_excluded == 0


def test_sheets_without_shared_invoices_exclude_nothing():
    rows = [
        make_row("1", "85123A", 1, 1.0, sheet="S1"),
        make_row("2", "85123A", 1, 1.0, sheet="S2", date="2020-01-05 10:00"),
    ]
    result = run(rows)
    assert result.sheets_in_order == ("S1", "S2")
    assert result.overlap_invoices_in_both == 0
    assert result.overlap_rows_excluded == 0
    assert result.annotated["reason_code"].tolist() == ["merchandise_sale"] * 2


def test_missing_customer_ids_in_both_sheets_count_as_identical():
    rows = [
        make_row("1", "85123A", 2, 1.0),
        make_row("2", "85123A", 1, 1.0, date=WINDOW, customer=None),
        make_row("2", "85123A", 1, 1.0, date=WINDOW, customer=None, sheet="S2"),
    ]
    assert reasons(rows) == [
        "merchandise_sale",
        "merchandise_sale",
        "sheet_overlap_copy",
    ]


# ====================== overlap: fail-safe when content differs ================
MISMATCH_VARIANTS = [
    pytest.param(
        [s2("85123A", 5, 1.0), s2("85123B", 1, 2.0)], id="quantity_differs"
    ),
    pytest.param(
        [s2("85123A", 1, 1.0), s2("85123B", 1, 2.5)], id="price_differs"
    ),
    pytest.param(
        [s2("85123A", 1, 1.0, description="OTHER TEXT"), s2("85123B", 1, 2.0)],
        id="description_differs",
    ),
    pytest.param(
        [s2("85123A", 1, 1.0, customer=2.0), s2("85123B", 1, 2.0)],
        id="customer_differs",
    ),
    pytest.param(
        [s2("85123A", 1, 1.0, date="2020-01-09 11:00"), s2("85123B", 1, 2.0)],
        id="date_differs",
    ),
    pytest.param(
        [s2("85123A", 1, 1.0), s2("85123B", 1, 2.0), s2("85123C", 1, 1.0)],
        id="extra_row_in_second",
    ),
    pytest.param([s2("85123A", 1, 1.0)], id="missing_row_in_second"),
    pytest.param(
        [s2("85123A", 1, 1.0), s2("85123A", 1, 1.0), s2("85123B", 1, 2.0)],
        id="repeated_row_count_differs",
    ),
]


@pytest.mark.parametrize("second_sheet_rows", MISMATCH_VARIANTS)
def test_overlap_mismatch_raises_and_removes_nothing(second_sheet_rows):
    frame = make_frame(s1_rows() + second_sheet_rows)
    before = frame.copy()
    with pytest.raises(OverlapMismatchError, match="Nothing was removed") as excinfo:
        clean_transactions(frame, make_config(), REQUIRED)
    error = excinfo.value
    assert (error.first_sheet, error.second_sheet) == ("S1", "S2")
    assert error.mismatched_count == 1
    assert error.examples == ("2",)
    pd.testing.assert_frame_equal(before, frame)


def test_mismatch_in_a_listed_code_row_still_raises():
    rows = [
        make_row("1", "85123A", 2, 1.0),
        make_row("7", "POST", 1, 1.0, date=WINDOW),
        make_row("7", "POST", 2, 1.0, date=WINDOW, sheet="S2"),
    ]
    with pytest.raises(OverlapMismatchError) as excinfo:
        run(rows)
    assert excinfo.value.examples == ("7",)


def test_missing_customer_id_in_only_one_sheet_is_a_mismatch():
    rows = [
        make_row("1", "85123A", 2, 1.0),
        make_row("2", "85123A", 1, 1.0, date=WINDOW, customer=None),
        make_row("2", "85123A", 1, 1.0, date=WINDOW, customer=1.0, sheet="S2"),
    ]
    with pytest.raises(OverlapMismatchError):
        run(rows)


def test_keep_all_keeps_both_copies_and_flags_the_repeats():
    result = run(identical_overlap_rows(), overlap_policy="keep_all")
    annotated = result.annotated
    assert annotated["reason_code"].tolist() == ["merchandise_sale"] * 6
    assert result.overlap_rows_excluded == 0
    assert result.overlap_invoices_in_both == 0
    assert result.sheets_in_order == ("S1", "S2")
    assert annotated["is_exact_duplicate"].tolist() == [
        False, False, False, True, True, False,
    ]


def test_keep_all_does_not_raise_when_overlap_content_differs():
    rows = s1_rows() + [s2("85123A", 5, 1.0), s2("85123B", 1, 2.0)]
    result = run(rows, overlap_policy="keep_all")
    assert result.annotated["reason_code"].tolist() == ["merchandise_sale"] * 5
    assert result.overlap_rows_excluded == 0


def _three_sheet_rows():
    return [
        make_row("1", "85123A", 1, 1.0, sheet="S1", date="2020-01-01 10:00"),
        make_row("2", "85123A", 1, 1.0, sheet="S2", date="2020-01-02 10:00"),
        make_row("3", "85123A", 1, 1.0, sheet="S3", date="2020-01-03 10:00"),
    ]


def test_three_sheets_with_keep_earlier_sheet_raises():
    with pytest.raises(CleaningError, match="at most two"):
        run(_three_sheet_rows())


def test_three_sheets_with_keep_all_works():
    result = run(_three_sheet_rows(), overlap_policy="keep_all")
    assert result.sheets_in_order == ("S1", "S2", "S3")
    assert result.annotated["reason_code"].tolist() == ["merchandise_sale"] * 3


# ====================== overlap: counting rows that are invalid ================
INVALID_OVERLAP_ROWS = [
    make_row("50", "85123A", 1, 1.0, date="2020-01-01 10:00"),
    make_row("100", "85123A", None, 1.0, date="2020-01-05 10:00"),
    make_row("100", "85123A", None, 1.0, date="2020-01-05 10:00", sheet="S2"),
]


def test_invalid_overlap_copy_gets_the_invalid_field_reason():
    assert reasons(INVALID_OVERLAP_ROWS) == [
        "merchandise_sale",
        "invalid_field",
        "invalid_field",
    ]


def test_overlap_rows_excluded_counts_only_rows_with_the_overlap_reason():
    result = run(INVALID_OVERLAP_ROWS)
    copies = int((result.annotated["reason_code"] == "sheet_overlap_copy").sum())
    assert result.overlap_rows_excluded == copies


# ================================== duplicates ================================
DUPLICATE_ROWS = [
    make_row("1", "85123A", 1, 1.0),
    make_row("1", "85123A", 1, 1.0),
    make_row("1", "85123A", 1, 1.0),
    make_row("1", "85123A", 1, 1.0, customer=2.0),
]


def test_duplicate_flag_marks_later_repeats_only():
    annotated = run(DUPLICATE_ROWS).annotated
    assert annotated["is_exact_duplicate"].tolist() == [False, True, True, False]
    assert annotated["reason_code"].tolist() == ["merchandise_sale"] * 4


def test_duplicate_drop_policy_excludes_later_repeats():
    annotated = run(DUPLICATE_ROWS, duplicate_policy="drop").annotated
    assert annotated["is_exact_duplicate"].tolist() == [False, True, True, False]
    assert annotated["reason_code"].tolist() == [
        "merchandise_sale",
        "exact_duplicate_dropped",
        "exact_duplicate_dropped",
        "merchandise_sale",
    ]


def test_duplicates_with_missing_values_match_each_other():
    rows = [
        make_row("1", "85123A", 1, 1.0, customer=None),
        make_row("1", "85123A", 1, 1.0, customer=None),
        make_row("2", "85123A", 1, 1.0),
    ]
    assert run(rows).annotated["is_exact_duplicate"].tolist() == [False, True, False]


# ============================== policy and code rules =========================
CANCELLATION_ROWS = [
    make_row("1", "85123A", 1, 1.0),
    make_row("C2", "85123A", -1, 1.0),
    make_row("C3", "gift_0001_20", -1, 20.0),
]


@pytest.mark.parametrize(
    "policy,expected",
    [
        pytest.param(
            "return",
            ["merchandise_sale", "merchandise_return", "voucher_return"],
            id="return",
        ),
        pytest.param(
            "exclude",
            ["merchandise_sale", "cancellation_excluded", "cancellation_excluded"],
            id="exclude",
        ),
    ],
)
def test_cancellation_policy(policy, expected):
    assert reasons(CANCELLATION_ROWS, cancellation_policy=policy) == expected


def test_voucher_prefix_matching_is_an_exact_case_sensitive_prefix():
    rows = [
        make_row("1", "gift_0001_20", 1, 5.0),
        make_row("2", "GIFT_0001_20", 1, 5.0),
        make_row("3", "xgift_0001_20", 1, 5.0),
        make_row("4", "gift_0002_20", 1, 5.0),
    ]
    result = run(rows)
    assert result.annotated["reason_code"].tolist() == [
        "voucher_sale",
        "merchandise_sale",
        "merchandise_sale",
        "merchandise_sale",
    ]
    assert result.annotated["is_voucher_code"].tolist() == [True, False, False, False]


def test_several_voucher_prefixes_are_all_recognised():
    rows = [
        make_row("1", "gift_0001_10", 1, 5.0),
        make_row("2", "VCH5", 1, 5.0),
        make_row("3", "85123A", 1, 5.0),
    ]
    assert reasons(rows, voucher_stock_code_prefixes=("gift_0001_", "VCH")) == [
        "voucher_sale",
        "voucher_sale",
        "merchandise_sale",
    ]


def test_no_voucher_prefixes_means_no_vouchers():
    result = run(
        [make_row("1", "gift_0001_20", 1, 20.0)], voucher_stock_code_prefixes=()
    )
    assert result.annotated["reason_code"].tolist() == ["merchandise_sale"]
    assert result.annotated["is_voucher_code"].tolist() == [False]


def test_stock_codes_are_not_trimmed_before_matching():
    rows = [
        make_row("1", " POST", 1, 1.0),
        make_row("2", "POST ", 1, 1.0),
        make_row("3", "POST", 1, 1.0),
    ]
    assert reasons(rows) == [
        "merchandise_sale",
        "merchandise_sale",
        "listed_non_product_code",
    ]


# ============================= unusual-code audit table ========================
AUDIT_ROWS = [
    make_row("1", "85123A", 1, 1.0),                                              # standard
    make_row("2", "15056BL", 1, 1.0),                                             # standard
    make_row("3", "DCGS0058", 1, 1.0, description="BOXED THING"),                 # uncovered
    make_row("4", "DCGS0058", 1, 0.0, description="BOXED THING"),                 # uncovered, zero price
    make_row("5", "PADS", 1, 1.0, description="PADS TO MATCH"),                   # uncovered
    make_row("6", "POST", 1, 1.0),                                                # listed
    make_row("7", "DCGSSGIRL", 1, 1.0),                                           # unidentifiable
    make_row("8", "gift_0001_20", 1, 20.0),                                       # voucher prefix
    make_row("9", " POST", 1, 1.0, description="SPACED"),                         # uncovered
    make_row("10", None, 1, 1.0),                                                 # missing code
]


def test_audit_lists_only_uncovered_unusual_codes():
    annotated = run(AUDIT_ROWS).annotated
    audit = audit_uncovered_unusual_codes(annotated, make_config())
    assert list(audit.columns) == [
        "StockCode",
        "rows",
        "retained_rows",
        "example_description",
    ]
    assert audit["StockCode"].tolist() == ["DCGS0058", " POST", "PADS"]
    assert audit["rows"].tolist() == [2, 1, 1]
    assert audit["retained_rows"].tolist() == [1, 1, 1]
    assert audit["example_description"].tolist() == [
        "BOXED THING",
        "SPACED",
        "PADS TO MATCH",
    ]


def test_unusual_code_format_alone_does_not_exclude_a_row():
    assert reasons(AUDIT_ROWS) == [
        "merchandise_sale",
        "merchandise_sale",
        "merchandise_sale",
        "zero_price",
        "merchandise_sale",
        "listed_non_product_code",
        "listed_unidentifiable_code",
        "voucher_sale",
        "merchandise_sale",
        "invalid_field",
    ]


def test_audit_is_empty_when_all_codes_are_standard_or_covered():
    rows = [make_row("1", "85123A", 1, 1.0), make_row("2", "15056BL", 1, 1.0)]
    audit = audit_uncovered_unusual_codes(run(rows).annotated, make_config())
    assert audit.empty
    assert list(audit.columns) == [
        "StockCode",
        "rows",
        "retained_rows",
        "example_description",
    ]


def test_audit_missing_column_raises():
    with pytest.raises(SchemaError, match="Description"):
        audit_uncovered_unusual_codes(
            pd.DataFrame({"StockCode": ["x"]}), make_config()
        )