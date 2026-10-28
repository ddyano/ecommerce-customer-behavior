"""Tests for the outputs of src.preprocessing.cleaning.

All data here is SYNTHETIC (invented rows). None of it comes from the real
Online Retail II dataset.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import CleaningConfig
from src.ingestion.loader import SchemaError
from src.preprocessing import rules
from src.preprocessing.cleaning import (
    ADDED_COLUMNS,
    ROLE_FLAG_COLUMNS,
    CleaningError,
    ReconciliationError,
    check_reconciliation,
    clean_transactions,
    format_reason_summary,
    select_basket_eligible,
    select_excluded,
    select_merchandise_returns,
    select_merchandise_sales,
    select_vouchers,
    summarize_reasons,
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
):
    return (invoice, code, "SYNTHETIC ITEM", quantity, date, price, customer, "UK", sheet)


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


# Eleven invented rows, one outcome each (hand-worked):
#  0 sale 2x3.0=6.0          1 sale 1x1.5 (no customer)   2 return -1x3.0=-3.0
#  3 voucher sale 20.0       4 voucher return -20.0       5 listed code POST 5.0
#  6 zero price 0.0          7 sale invoice qty -2 => -4.0 8 negative price -1.0
#  9 missing stock code 1.0  10 unreadable price (no line value)
MIXED_ROWS = [
    make_row("1001", "85123A", 2, 3.0),
    make_row("1002", "85123B", 1, 1.5, customer=None),
    make_row("C1003", "85123A", -1, 3.0),
    make_row("1004", "gift_0001_20", 1, 20.0),
    make_row("C1005", "gift_0001_20", -1, 20.0),
    make_row("1006", "POST", 1, 5.0),
    make_row("1007", "85123A", 1, 0.0),
    make_row("1008", "85123A", -2, 2.0),
    make_row("1009", "85123A", 1, -1.0),
    make_row("1010", None, 1, 1.0),
    make_row("1011", "85123A", 1, "abc"),
]
N_ROWS = len(MIXED_ROWS)

EXPECTED_REASONS = [
    "merchandise_sale",
    "merchandise_sale",
    "merchandise_return",
    "voucher_sale",
    "voucher_return",
    "listed_non_product_code",
    "zero_price",
    "nonpositive_quantity",
    "negative_price",
    "invalid_field",
    "invalid_field",
]
EXPECTED_ROLES = (
    ["merchandise_sale", "merchandise_sale", "merchandise_return"]
    + ["voucher", "voucher"]
    + ["excluded"] * 6
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return make_frame(MIXED_ROWS)


@pytest.fixture()
def result(frame):
    return clean_transactions(frame, make_config(), REQUIRED)


@pytest.fixture()
def annotated(result) -> pd.DataFrame:
    return result.annotated


# ------------------------- shape and raw-data preservation ---------------------
def test_result_has_same_row_count_and_all_added_columns(frame, annotated):
    assert len(annotated) == len(frame) == N_ROWS
    assert set(ADDED_COLUMNS) <= set(annotated.columns)
    assert annotated["row_id"].tolist() == list(range(N_ROWS))
    assert annotated["reason_code"].tolist() == EXPECTED_REASONS


def test_raw_columns_are_unchanged(frame, annotated):
    pd.testing.assert_frame_equal(annotated[COLUMNS], frame)


def test_input_frame_is_not_modified(frame):
    before = frame.copy()
    clean_transactions(frame, make_config(), REQUIRED)
    pd.testing.assert_frame_equal(before, frame)


def test_non_default_index_is_reset(frame):
    shifted = frame.copy()
    shifted.index = pd.Index(range(100, 100 + N_ROWS))
    cleaned = clean_transactions(shifted, make_config(), REQUIRED).annotated
    pd.testing.assert_frame_equal(cleaned[COLUMNS], frame.reset_index(drop=True))
    assert cleaned["row_id"].tolist() == list(range(N_ROWS))


def test_parsed_columns_hold_expected_values(annotated):
    np.testing.assert_allclose(
        annotated["line_value"].to_numpy(dtype=float),
        [6.0, 1.5, -3.0, 20.0, -20.0, 5.0, 0.0, -4.0, -1.0, 1.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        annotated["quantity_num"].to_numpy(dtype=float),
        [2, 1, -1, 1, -1, 1, 1, -2, 1, 1, 1],
    )
    np.testing.assert_allclose(
        annotated["price_num"].to_numpy(dtype=float),
        [3.0, 1.5, 3.0, 20.0, 20.0, 5.0, 0.0, 2.0, -1.0, 1.0, np.nan],
        equal_nan=True,
    )
    assert annotated["invoice_datetime"].notna().all()


def test_raw_text_is_kept_when_parsing_fails(annotated):
    assert annotated.loc[10, "Price"] == "abc"
    assert pd.isna(annotated.loc[10, "price_num"])
    assert annotated.loc[10, "reason_code"] == "invalid_field"


def test_customer_cancellation_and_voucher_flags(annotated):
    assert annotated["has_customer_id"].tolist() == [i != 1 for i in range(N_ROWS)]
    assert annotated["is_cancellation_invoice"].tolist() == [
        i in {2, 4} for i in range(N_ROWS)
    ]
    assert annotated["is_voucher_code"].tolist() == [i in {3, 4} for i in range(N_ROWS)]


def test_cleaning_is_deterministic(frame):
    first = clean_transactions(frame, make_config(), REQUIRED).annotated
    second = clean_transactions(frame, make_config(), REQUIRED).annotated
    pd.testing.assert_frame_equal(first, second)


# ------------------------------- roles and flags ------------------------------
def test_exactly_one_role_flag_per_row(annotated):
    assert annotated[list(ROLE_FLAG_COLUMNS)].sum(axis=1).tolist() == [1] * N_ROWS


def test_roles_and_dispositions_follow_reasons(annotated):
    assert annotated["analytic_role"].tolist() == EXPECTED_ROLES
    assert annotated["disposition"].tolist() == ["retained"] * 5 + ["excluded"] * 6


def test_basket_flag_only_for_merchandise_sales(annotated):
    assert annotated["is_basket_eligible"].tolist() == [True, True] + [False] * 9


def test_flag_columns_are_boolean(annotated):
    boolean_columns = (
        *ROLE_FLAG_COLUMNS,
        "is_basket_eligible",
        "is_exact_duplicate",
        "has_customer_id",
        "is_cancellation_invoice",
        "is_voucher_code",
    )
    for column in boolean_columns:
        assert pd.api.types.is_bool_dtype(annotated[column]), column


# ---------------------------------- selectors ---------------------------------
ALL_SELECTORS = [
    select_merchandise_sales,
    select_merchandise_returns,
    select_vouchers,
    select_excluded,
    select_basket_eligible,
]


def test_selector_row_ids(annotated):
    assert select_merchandise_sales(annotated)["row_id"].tolist() == [0, 1]
    assert select_merchandise_returns(annotated)["row_id"].tolist() == [2]
    assert select_vouchers(annotated)["row_id"].tolist() == [3, 4]
    assert select_excluded(annotated)["row_id"].tolist() == [5, 6, 7, 8, 9, 10]
    assert select_basket_eligible(annotated)["row_id"].tolist() == [0, 1]


def test_selectors_partition_the_rows(annotated):
    parts = [
        select_merchandise_sales(annotated),
        select_merchandise_returns(annotated),
        select_vouchers(annotated),
        select_excluded(annotated),
    ]
    id_sets = [set(part["row_id"]) for part in parts]
    assert sum(len(ids) for ids in id_sets) == len(annotated)  # no overlap
    assert set().union(*id_sets) == set(annotated["row_id"])  # nothing missing


def test_vouchers_never_reach_merchandise_or_basket(annotated):
    voucher_ids = set(select_vouchers(annotated)["row_id"])
    assert voucher_ids == {3, 4}
    merchandise_ids = (
        set(select_merchandise_sales(annotated)["row_id"])
        | set(select_merchandise_returns(annotated)["row_id"])
        | set(select_basket_eligible(annotated)["row_id"])
    )
    assert voucher_ids.isdisjoint(merchandise_ids)


def test_basket_selector_matches_sales_selector(annotated):
    assert (
        select_basket_eligible(annotated)["row_id"].tolist()
        == select_merchandise_sales(annotated)["row_id"].tolist()
    )


def test_selectors_return_rows_unmodified(annotated):
    picked = select_vouchers(annotated)
    pd.testing.assert_frame_equal(picked, annotated.loc[picked.index])
    assert list(picked.columns) == list(annotated.columns)


def test_selector_with_no_matches_returns_empty_frame():
    only_sale = make_frame([make_row("1", "85123A", 1, 1.0)])
    cleaned = clean_transactions(only_sale, make_config(), REQUIRED).annotated
    empty = select_vouchers(cleaned)
    assert len(empty) == 0
    assert list(empty.columns) == list(cleaned.columns)


@pytest.mark.parametrize("selector", ALL_SELECTORS, ids=lambda fn: fn.__name__)
def test_selector_missing_column_raises(selector):
    with pytest.raises(SchemaError):
        selector(pd.DataFrame({"a": [1]}))


# -------------------------------- reconciliation ------------------------------
def test_reconciliation_returns_counts(annotated):
    assert check_reconciliation(annotated, N_ROWS) == {
        "total_rows": N_ROWS,
        "retained_rows": 5,
        "excluded_rows": 6,
    }


def _duplicate_row_id(d):
    d.loc[1, "row_id"] = 0


def _reversed_row_id(d):
    d["row_id"] = d["row_id"].to_numpy()[::-1].copy()


def _wrong_disposition(d):
    d.loc[0, "disposition"] = "excluded"


def _wrong_role(d):
    d.loc[0, "analytic_role"] = "voucher"


def _extra_voucher_flag(d):
    d.loc[0, "is_voucher"] = True


def _wrong_basket_flag(d):
    d.loc[2, "is_basket_eligible"] = True


def _unknown_reason(d):
    d.loc[0, "reason_code"] = "mystery"


@pytest.mark.parametrize(
    "mutate,message",
    [
        pytest.param(_duplicate_row_id, "row_id", id="duplicate_row_id"),
        pytest.param(_reversed_row_id, "row_id", id="reversed_row_id"),
        pytest.param(_wrong_disposition, "disposition does not match", id="wrong_disposition"),
        pytest.param(_wrong_role, "analytic_role does not match", id="wrong_role"),
        pytest.param(_extra_voucher_flag, "is_voucher does not match", id="extra_voucher_flag"),
        pytest.param(_wrong_basket_flag, "is_basket_eligible", id="wrong_basket_flag"),
        pytest.param(_unknown_reason, "Unknown", id="unknown_reason"),
    ],
)
def test_reconciliation_detects_tampering(annotated, mutate, message):
    tampered = annotated.copy()
    mutate(tampered)
    with pytest.raises(ReconciliationError, match=message):
        check_reconciliation(tampered, N_ROWS)


def test_reconciliation_detects_wrong_raw_row_count(annotated):
    with pytest.raises(ReconciliationError, match="row count"):
        check_reconciliation(annotated, N_ROWS + 1)


def test_reconciliation_detects_missing_column(annotated):
    with pytest.raises(ReconciliationError, match="missing column"):
        check_reconciliation(annotated.drop(columns=["row_id"]), N_ROWS)


# ------------------------------- reason summary -------------------------------
EXPECTED_SUMMARY = {
    # reason: (rows, units, line_value)
    "merchandise_sale": (2, 3.0, 7.5),
    "merchandise_return": (1, -1.0, -3.0),
    "voucher_sale": (1, 1.0, 20.0),
    "voucher_return": (1, -1.0, -20.0),
    "listed_non_product_code": (1, 1.0, 5.0),
    "zero_price": (1, 1.0, 0.0),
    "nonpositive_quantity": (1, -2.0, -4.0),
    "negative_price": (1, 1.0, -1.0),
    "invalid_field": (2, 2.0, 1.0),
}


def test_reason_summary_order_and_columns(result):
    summary = result.reason_summary
    assert summary["reason_code"].tolist() == list(rules.REASON_CODES)
    assert list(summary.columns) == [
        "reason_code",
        "disposition",
        "role",
        "rows",
        "units",
        "line_value",
        "rows_without_value",
    ]


def test_reason_summary_totals(result, annotated):
    summary = result.reason_summary
    assert int(summary["rows"].sum()) == N_ROWS
    assert float(summary["line_value"].sum()) == pytest.approx(5.5)
    assert float(summary["line_value"].sum()) == pytest.approx(
        float(annotated["line_value"].sum())
    )
    assert int(summary["rows_without_value"].sum()) == 1


def test_reason_summary_values_per_reason(result):
    by_reason = result.reason_summary.set_index("reason_code")
    for code in rules.REASON_CODES:
        rows, units, value = EXPECTED_SUMMARY.get(code, (0, 0.0, 0.0))
        assert int(by_reason.loc[code, "rows"]) == rows, code
        assert float(by_reason.loc[code, "units"]) == pytest.approx(units), code
        assert float(by_reason.loc[code, "line_value"]) == pytest.approx(value), code
    assert int(by_reason.loc["invalid_field", "rows_without_value"]) == 1


def test_reason_summary_dispositions_and_roles(result):
    for _, row in result.reason_summary.iterrows():
        assert row["disposition"] == rules.DISPOSITION_BY_CODE[row["reason_code"]]
        assert row["role"] == rules.ROLE_BY_CODE[row["reason_code"]]


def test_summarize_reasons_missing_column_raises():
    with pytest.raises(SchemaError, match="quantity_num"):
        summarize_reasons(pd.DataFrame({"reason_code": ["x"]}))


def test_format_reason_summary_text(result):
    text = format_reason_summary(result.reason_summary)
    assert "Total rows: 11 (retained 5, excluded 6)" in text
    assert "merchandise_sale" in text


# ------------------------------- input validation -----------------------------
def test_empty_input_produces_empty_annotated_table():
    cleaned = clean_transactions(make_frame([]), make_config(), REQUIRED)
    assert len(cleaned.annotated) == 0
    assert set(ADDED_COLUMNS) <= set(cleaned.annotated.columns)
    assert len(cleaned.reason_summary) == len(rules.REASON_CODES)
    assert int(cleaned.reason_summary["rows"].sum()) == 0
    assert check_reconciliation(cleaned.annotated, 0) == {
        "total_rows": 0,
        "retained_rows": 0,
        "excluded_rows": 0,
    }


def test_missing_core_column_raises(frame):
    with pytest.raises(SchemaError, match="Price"):
        clean_transactions(frame.drop(columns=["Price"]), make_config(), REQUIRED)


def test_reserved_column_raises(frame):
    frame["row_id"] = 0
    with pytest.raises(CleaningError, match="reserved"):
        clean_transactions(frame, make_config(), REQUIRED)


def test_missing_source_sheet_value_raises(frame):
    frame.loc[0, "source_sheet"] = None
    with pytest.raises(CleaningError, match="source_sheet"):
        clean_transactions(frame, make_config(), REQUIRED)