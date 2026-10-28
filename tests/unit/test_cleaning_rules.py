"""Tests for the cleaning rule vocabulary and first-match precedence.

All data here is SYNTHETIC (invented rows). None of it comes from the real
Online Retail II dataset.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.config import CleaningConfig
from src.preprocessing import rules
from src.preprocessing.cleaning import clean_transactions

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
    return (
        invoice,
        code,
        "SYNTHETIC ITEM",
        quantity,
        date,
        price,
        customer,
        "UK",
        sheet,
    )


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


def reasons_for(rows, **overrides) -> list[str]:
    result = clean_transactions(make_frame(rows), make_config(**overrides), REQUIRED)
    return result.annotated["reason_code"].tolist()


# ----------------------------- rule vocabulary -------------------------------
EXPECTED_ORDER = [
    "invalid_field",
    "sheet_overlap_copy",
    "exact_duplicate_dropped",
    "listed_non_product_code",
    "listed_unidentifiable_code",
    "unrecognized_invoice_prefix",
    "negative_price",
    "zero_price",
    "cancellation_excluded",
    "cancellation_nonnegative_quantity",
    "nonpositive_quantity",
    "voucher_return",
    "voucher_sale",
    "merchandise_return",
    "merchandise_sale",
]


def test_reason_order_is_the_documented_precedence():
    assert list(rules.REASON_CODES) == EXPECTED_ORDER


def test_reason_codes_are_unique_and_complete():
    assert len(rules.REASON_CODES) == 15
    assert len(set(rules.REASON_CODES)) == 15


def test_lookup_tables_cover_exactly_the_reason_codes():
    codes = set(rules.REASON_CODES)
    assert set(rules.REASON_BY_CODE) == codes
    assert set(rules.DISPOSITION_BY_CODE) == codes
    assert set(rules.ROLE_BY_CODE) == codes


def test_only_expected_reasons_are_retained():
    retained = {
        code
        for code, disposition in rules.DISPOSITION_BY_CODE.items()
        if disposition == "retained"
    }
    assert retained == {
        "voucher_return",
        "voucher_sale",
        "merchandise_return",
        "merchandise_sale",
    }


def test_roles_match_reasons():
    retained_roles = {
        "voucher_return": "voucher",
        "voucher_sale": "voucher",
        "merchandise_return": "merchandise_return",
        "merchandise_sale": "merchandise_sale",
    }
    for code, role in retained_roles.items():
        assert rules.ROLE_BY_CODE[code] == role
    for code in rules.REASON_CODES:
        if code not in retained_roles:
            assert rules.ROLE_BY_CODE[code] == "excluded"


def test_dispositions_and_roles_use_known_values():
    assert set(rules.DISPOSITION_BY_CODE.values()) == {"retained", "excluded"}
    assert set(rules.ROLE_BY_CODE.values()) == set(rules.ROLES)


def test_every_reason_has_a_description():
    for info in rules.REASONS:
        assert info.description.strip(), f"{info.code} has no description"


def test_reason_wording_is_neutral():
    """Codes and descriptions must describe conditions, not guess at causes."""
    guesses = ("debt", "adjust", "fraud", "mistake")
    for info in rules.REASONS:
        text = f"{info.code} {info.description}".lower()
        assert not any(word in text for word in guesses), info.code


# ------------------- precedence: one row matching several rules ----------------
SINGLE_ROW_CASES = [
    pytest.param(
        make_row("C1001", "POST", -1, 0.0), {}, "listed_non_product_code",
        id="listed_code_beats_cancellation_and_zero_price",
    ),
    pytest.param(
        make_row("1002", "gift_0001_20", 1, 0.0), {}, "zero_price",
        id="voucher_sale_with_zero_price_is_zero_price",
    ),
    pytest.param(
        make_row("C1003", "gift_0001_20", -1, -1.0), {}, "negative_price",
        id="voucher_cancellation_with_negative_price_is_negative_price",
    ),
    pytest.param(
        make_row("C1004", "gift_0001_20", -1, 0.0),
        {"zero_price_policy": "keep"}, "voucher_return",
        id="voucher_cancellation_with_zero_price_kept_is_voucher_return",
    ),
    pytest.param(
        make_row("C1005", "gift_0001_20", -1, 2.0),
        {"cancellation_policy": "exclude"}, "cancellation_excluded",
        id="voucher_cancellation_excluded_by_policy",
    ),
    pytest.param(
        make_row("C1006", "85123A", 2, 0.0), {}, "zero_price",
        id="cancellation_positive_quantity_zero_price_is_zero_price",
    ),
    pytest.param(
        make_row("A1007", "85123A", 1, 0.0), {}, "unrecognized_invoice_prefix",
        id="unrecognized_prefix_beats_zero_price",
    ),
    pytest.param(
        make_row("A1008", "B", 1, 1.0), {}, "listed_non_product_code",
        id="listed_code_beats_unrecognized_prefix",
    ),
    pytest.param(
        make_row("1009", "85123A", 0, 0.0), {}, "zero_price",
        id="zero_quantity_zero_price_is_zero_price",
    ),
    pytest.param(
        make_row("1010", "85123A", -1, -1.0), {}, "negative_price",
        id="negative_quantity_negative_price_is_negative_price",
    ),
    pytest.param(
        make_row("C1011", "DCGSSGIRL", -1, 1.0), {}, "listed_unidentifiable_code",
        id="unidentifiable_code_on_a_cancellation",
    ),
    pytest.param(
        make_row("C1012", "85123A", -1, 2.0), {}, "merchandise_return",
        id="valid_return",
    ),
    pytest.param(
        make_row("1013", "85123A", 2, 2.0), {}, "merchandise_sale",
        id="valid_sale",
    ),
    pytest.param(
        make_row("1014", "gift_0001_20", 1, 20.0), {}, "voucher_sale",
        id="valid_voucher_sale",
    ),
    pytest.param(
        make_row("C1015", "gift_0001_20", 1, 20.0), {},
        "cancellation_nonnegative_quantity",
        id="voucher_cancellation_with_positive_quantity",
    ),
    pytest.param(
        make_row("1016", "85123A", 1, 0.0), {"zero_price_policy": "keep"},
        "merchandise_sale",
        id="zero_price_kept_by_policy_is_a_sale",
    ),
    pytest.param(
        make_row("1017", "85123A", -2, 2.0), {}, "nonpositive_quantity",
        id="sale_invoice_with_negative_quantity",
    ),
    pytest.param(
        make_row("C1018", "85123A", -1, -2.0), {}, "negative_price",
        id="negative_price_on_a_cancellation",
    ),
    pytest.param(
        make_row("C1019", "85123A", -1, 0.0),
        {"cancellation_policy": "exclude"}, "zero_price",
        id="zero_price_beats_cancellation_policy_exclude",
    ),
    pytest.param(
        make_row("1020", None, 1, 1.0), {}, "invalid_field",
        id="invalid_missing_stock_code",
    ),
    pytest.param(
        make_row("1021", "POST", 1, 1.0, date=None), {}, "invalid_field",
        id="invalid_beats_listed_code",
    ),
    pytest.param(
        make_row(None, "85123A", 1, 1.0), {}, "invalid_field",
        id="invalid_missing_invoice",
    ),
    pytest.param(
        make_row("1022", "85123A", 1, "abc"), {}, "invalid_field",
        id="invalid_unreadable_price",
    ),
    pytest.param(
        make_row("1023", "85123A", "x", 1.0), {}, "invalid_field",
        id="invalid_unreadable_quantity",
    ),
    pytest.param(
        make_row("c1024", "85123A", -1, 1.0), {}, "unrecognized_invoice_prefix",
        id="lowercase_c_invoice_is_not_a_cancellation",
    ),
]


@pytest.mark.parametrize("row,overrides,expected", SINGLE_ROW_CASES)
def test_first_match_reason_for_a_single_row(row, overrides, expected):
    assert reasons_for([row], **overrides) == [expected]


# ------------------ precedence: duplicates, prefix, case, overlap ---------------
def test_duplicate_drop_policy_beats_listed_code():
    rows = [make_row("5", "POST", 1, 1.0), make_row("5", "POST", 1, 1.0)]
    assert reasons_for(rows, duplicate_policy="drop") == [
        "listed_non_product_code",
        "exact_duplicate_dropped",
    ]


def test_duplicate_flag_policy_keeps_the_normal_reason():
    rows = [make_row("5", "POST", 1, 1.0), make_row("5", "POST", 1, 1.0)]
    result = clean_transactions(make_frame(rows), make_config(), REQUIRED)
    assert result.annotated["reason_code"].tolist() == [
        "listed_non_product_code",
        "listed_non_product_code",
    ]
    assert result.annotated["is_exact_duplicate"].tolist() == [False, True]


def test_custom_cancellation_prefix_changes_what_is_a_cancellation():
    rows = [make_row("X1001", "85123A", -1, 1.0), make_row("C1002", "85123A", -1, 1.0)]
    assert reasons_for(rows, cancellation_invoice_prefix="X") == [
        "merchandise_return",
        "unrecognized_invoice_prefix",
    ]


def test_stock_code_matching_is_case_sensitive():
    rows = [make_row("1", "M", 1, 1.0), make_row("2", "m", 1, 1.0)]
    assert reasons_for(rows, non_product_stock_codes=("M",)) == [
        "listed_non_product_code",
        "merchandise_sale",
    ]


def test_overlap_copy_of_a_listed_code_is_an_overlap_copy():
    rows = [
        make_row("50", "85123A", 1, 1.0, date="2020-01-01 10:00"),
        make_row("100", "POST", 1, 1.0, date="2020-01-05 10:00"),
        make_row("100", "POST", 1, 1.0, date="2020-01-05 10:00", sheet="S2"),
    ]
    result = clean_transactions(make_frame(rows), make_config(), REQUIRED)
    assert result.annotated["reason_code"].tolist() == [
        "merchandise_sale",
        "listed_non_product_code",
        "sheet_overlap_copy",
    ]
    assert result.sheets_in_order == ("S1", "S2")
    assert result.overlap_invoices_in_both == 1


def test_invalid_field_beats_overlap_copy():
    rows = [
        make_row("50", "85123A", 1, 1.0, date="2020-01-01 10:00"),
        make_row("100", "85123A", None, 1.0, date="2020-01-05 10:00"),
        make_row("100", "85123A", None, 1.0, date="2020-01-05 10:00", sheet="S2"),
    ]
    assert reasons_for(rows) == ["merchandise_sale", "invalid_field", "invalid_field"]


# ------------------------ every reason can be reached --------------------------
def test_every_reason_code_is_reachable():
    rows = [
        make_row("1", "85123A", 1, 1.0),                                    # 0
        make_row("C2", "85123A", -1, 1.0),                                  # 1
        make_row("3", "gift_0001_20", 1, 20.0),                             # 2
        make_row("C4", "gift_0001_20", -1, 20.0),                           # 3
        make_row("5", "POST", 1, 1.0),                                      # 4
        make_row("6", "DCGSSGIRL", 1, 1.0),                                 # 5
        make_row("A7", "85123A", 1, 1.0),                                   # 6
        make_row("8", "85123A", 1, -1.0),                                   # 7
        make_row("9", "85123A", 1, 0.0),                                    # 8
        make_row("C10", "85123A", 2, 1.0),                                  # 9
        make_row("11", "85123A", -1, 1.0),                                  # 10
        make_row("12", None, 1, 1.0),                                       # 11
        make_row("1", "85123A", 1, 1.0),                                    # 12
        make_row("13", "85123A", 1, 1.0, date="2020-01-02 10:00"),          # 13
        make_row("13", "85123A", 1, 1.0, date="2020-01-02 10:00", sheet="S2"),  # 14
    ]
    expected_first_run = [
        "merchandise_sale",
        "merchandise_return",
        "voucher_sale",
        "voucher_return",
        "listed_non_product_code",
        "listed_unidentifiable_code",
        "unrecognized_invoice_prefix",
        "negative_price",
        "zero_price",
        "cancellation_nonnegative_quantity",
        "nonpositive_quantity",
        "invalid_field",
        "exact_duplicate_dropped",
        "merchandise_sale",
        "sheet_overlap_copy",
    ]
    first_run = reasons_for(rows, duplicate_policy="drop")
    assert first_run == expected_first_run

    second_run = reasons_for(
        [make_row("C2", "85123A", -1, 1.0)], cancellation_policy="exclude"
    )
    assert second_run == ["cancellation_excluded"]

    assert set(first_run) | set(second_run) == set(rules.REASON_CODES)