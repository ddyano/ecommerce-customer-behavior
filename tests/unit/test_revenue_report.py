"""Tests for src.reporting.revenue.

All data here is SYNTHETIC (invented rows with hand-computed expected answers).
None of it comes from the real Online Retail II dataset.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import CleaningConfig
from src.ingestion.loader import SchemaError
from src.preprocessing.cleaning import ReconciliationError, clean_transactions
from src.reporting.revenue import (
    REVENUE_TOLERANCE,
    RevenueError,
    RevenueReconciliationError,
    build_revenue_report,
    format_revenue_report,
    verify_revenue_report,
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


def annotate(rows, **overrides) -> pd.DataFrame:
    """Run the real cleaning step on invented rows and return the annotated table."""
    return clean_transactions(make_frame(rows), make_config(**overrides), REQUIRED).annotated


def assert_totals(totals, rows, invoices, units, value):
    assert totals.rows == rows
    assert totals.invoices == invoices
    assert totals.units == pytest.approx(units, abs=1e-9)
    assert totals.value == pytest.approx(value, abs=1e-9)


# Twenty-six invented rows, one outcome each (hand-worked; row number = row_id):
MIXED_ROWS = [
    make_row("1001", "85123A", 2, 3.0),                                    # 0  sale 6.0
    make_row("1001", "85123B", 1, 1.5),                                    # 1  sale 1.5
    make_row("1002", "85123A", 4, 2.5),                                    # 2  sale 10.0
    make_row("1002", "85123A", 4, 2.5),                                    # 3  sale 10.0, repeat of row 2
    make_row("C1003", "85123A", -1, 3.0),                                  # 4  return -3.0
    make_row("C1004", "85123B", -2, 1.5),                                  # 5  return -3.0
    make_row("C1004", "85123B", -2, 1.5),                                  # 6  return -3.0, repeat of row 5
    make_row("1005", "gift_0001_20", 1, 20.0),                             # 7  voucher sale 20.0
    make_row("1005", "gift_0001_20", 1, 20.0),                             # 8  voucher sale, repeat of row 7
    make_row("C1006", "gift_0001_20", -1, 20.0),                           # 9  voucher return -20.0
    make_row("1007", "POST", 1, 5.0, description="POSTAGE"),               # 10 listed code 5.0
    make_row("1008", "POST", 2, 5.0, description="POSTAGE"),               # 11 listed code 10.0
    make_row("C1009", "M", -1, 8.0, description="Manual"),                 # 12 listed code -8.0
    make_row("1010", "DCGSSGIRL", 3, 2.0),                                 # 13 unidentifiable 6.0
    make_row("1011", "85123A", 5, 0.0, description=None, customer=None),   # 14 zero price, qty > 0
    make_row("1012", "85123B", -4, 0.0, description="damaged", customer=None),  # 15 zero price, qty < 0
    make_row("1013", "85123A", 0, 0.0),                                    # 16 zero price, qty = 0
    make_row("1014", "85123A", 2, 0.0, description=None, customer=7.0),    # 17 zero price, qty > 0
    make_row("1015", "85123A", 1, "abc"),                                  # 18 invalid, no value
    make_row("1016", "85123A", "x", 2.0),                                  # 19 invalid, no value
    make_row("1017", None, 2, 4.0),                                        # 20 invalid, value 8.0
    make_row("C1018", "85123A", 2, 1.0),                                   # 21 cancellation, qty >= 0
    make_row("1019", "85123A", -3, 2.0),                                   # 22 sale invoice, qty <= 0
    make_row("1020", "85123A", 1, -1.5),                                   # 23 negative price
    make_row("A1021", "85123A", 1, 1.0),                                   # 24 unrecognized prefix
    make_row("1007", "POST", 1, 5.0, description="POSTAGE"),               # 25 listed code, repeat of row 10
]
EXPECTED_REASONS = (
    ["merchandise_sale"] * 4
    + ["merchandise_return"] * 3
    + ["voucher_sale"] * 2
    + ["voucher_return"]
    + ["listed_non_product_code"] * 3
    + ["listed_unidentifiable_code"]
    + ["zero_price"] * 4
    + ["invalid_field"] * 3
    + [
        "cancellation_nonnegative_quantity",
        "nonpositive_quantity",
        "negative_price",
        "unrecognized_invoice_prefix",
        "listed_non_product_code",
    ]
)

# reason: (rows, units, line value, rows without a value), in precedence order
EXPECTED_EXCLUDED = {
    "invalid_field": (3, 3.0, 8.0, 2),
    "sheet_overlap_copy": (0, 0.0, 0.0, 0),
    "exact_duplicate_dropped": (0, 0.0, 0.0, 0),
    "listed_non_product_code": (4, 3.0, 12.0, 0),
    "listed_unidentifiable_code": (1, 3.0, 6.0, 0),
    "unrecognized_invoice_prefix": (1, 1.0, 1.0, 0),
    "negative_price": (1, 1.0, -1.5, 0),
    "zero_price": (4, 3.0, 0.0, 0),
    "cancellation_excluded": (0, 0.0, 0.0, 0),
    "cancellation_nonnegative_quantity": (1, 2.0, 2.0, 0),
    "nonpositive_quantity": (1, -3.0, -6.0, 0),
}


@pytest.fixture()
def annotated() -> pd.DataFrame:
    return annotate(MIXED_ROWS)


@pytest.fixture()
def report(annotated):
    return build_revenue_report(annotated)


# ------------------------------ fixture sanity --------------------------------
def test_fixture_rows_get_the_intended_reasons_and_flags(annotated):
    assert len(EXPECTED_REASONS) == len(MIXED_ROWS) == 26
    assert annotated["reason_code"].tolist() == EXPECTED_REASONS
    flagged = annotated.loc[annotated["is_exact_duplicate"], "row_id"].tolist()
    assert flagged == [3, 6, 8, 25]
    assert int(annotated["line_value"].isna().sum()) == 2


# ------------------------------ section totals --------------------------------
def test_merchandise_totals(report):
    assert_totals(report.merchandise_sales, rows=4, invoices=2, units=11.0, value=27.5)
    assert_totals(report.merchandise_returns, rows=3, invoices=2, units=-5.0, value=-9.0)
    assert_totals(report.net_merchandise, rows=7, invoices=4, units=6.0, value=18.5)


def test_voucher_totals_are_separate_from_merchandise(report):
    assert_totals(report.voucher_sales, rows=2, invoices=1, units=2.0, value=40.0)
    assert_totals(report.voucher_returns, rows=1, invoices=1, units=-1.0, value=-20.0)
    assert_totals(report.voucher_net, rows=3, invoices=2, units=1.0, value=20.0)
    # Vouchers are not inside the merchandise totals.
    assert report.net_merchandise.value == pytest.approx(
        report.merchandise_sales.value + report.merchandise_returns.value
    )


def test_excluded_total(report):
    assert_totals(report.excluded_total, rows=16, invoices=15, units=13.0, value=21.5)


def test_totals_reconcile_with_the_annotated_table(annotated, report):
    assert report.total_rows == len(annotated) == 26
    total_value = float(annotated["line_value"].sum())
    total_units = float(annotated["quantity_num"].sum())
    assert total_value == pytest.approx(60.0, abs=1e-9)
    assert total_units == pytest.approx(20.0, abs=1e-9)
    sections_value = (
        report.net_merchandise.value + report.voucher_net.value + report.excluded_total.value
    )
    sections_units = (
        report.net_merchandise.units + report.voucher_net.units + report.excluded_total.units
    )
    assert sections_value == pytest.approx(total_value, abs=0.01)
    assert sections_units == pytest.approx(total_units, abs=0.01)
    sections_rows = (
        report.merchandise_sales.rows
        + report.merchandise_returns.rows
        + report.voucher_net.rows
        + report.excluded_total.rows
    )
    assert sections_rows == report.total_rows


def test_all_reconciliation_checks_pass(report):
    assert report.tolerance == REVENUE_TOLERANCE == 0.01
    assert len(report.checks) == 12
    assert all(check.passed for check in report.checks)
    names = [check.name for check in report.checks]
    assert len(set(names)) == len(names)


def test_building_the_report_does_not_modify_the_input(annotated):
    before = annotated.copy()
    build_revenue_report(annotated)
    pd.testing.assert_frame_equal(before, annotated)


def test_custom_tolerance_is_recorded(annotated):
    assert build_revenue_report(annotated, tolerance=0.5).tolerance == 0.5


# ------------------------- rows without a computable value --------------------
def test_rows_without_a_value_are_counted_and_stay_excluded(annotated, report):
    by_reason = report.excluded_by_reason.set_index("reason_code")
    assert int(by_reason.loc["invalid_field", "rows_without_value"]) == 2
    assert int(report.excluded_by_reason["rows_without_value"].sum()) == 2
    missing = annotated[annotated["line_value"].isna()]
    assert missing["reason_code"].tolist() == ["invalid_field", "invalid_field"]
    assert bool(missing["is_excluded"].all())
    # A readable quantity is counted in units; an unreadable one adds nothing.
    assert float(by_reason.loc["invalid_field", "units"]) == pytest.approx(3.0)
    assert float(by_reason.loc["invalid_field", "line_value"]) == pytest.approx(8.0)
    assert report.excluded_total.units == pytest.approx(13.0)


def test_retained_row_without_a_value_is_rejected(annotated):
    tampered = annotated.copy()
    tampered.loc[0, "line_value"] = np.nan
    with pytest.raises(
        RevenueReconciliationError, match="every retained row has a computable line value"
    ):
        build_revenue_report(tampered)


# --------------------------------- breakdowns ---------------------------------
def test_excluded_by_reason_table(report):
    table = report.excluded_by_reason
    assert list(table.columns) == [
        "reason_code",
        "rows",
        "units",
        "line_value",
        "rows_without_value",
    ]
    assert table["reason_code"].tolist() == list(EXPECTED_EXCLUDED)
    by_reason = table.set_index("reason_code")
    for code, (rows, units, value, missing) in EXPECTED_EXCLUDED.items():
        assert int(by_reason.loc[code, "rows"]) == rows, code
        assert float(by_reason.loc[code, "units"]) == pytest.approx(units), code
        assert float(by_reason.loc[code, "line_value"]) == pytest.approx(value), code
        assert int(by_reason.loc[code, "rows_without_value"]) == missing, code
    assert int(table["rows"].sum()) == report.excluded_total.rows == 16


def test_listed_code_values(report):
    table = report.listed_code_values
    assert list(table.columns) == [
        "reason_code",
        "StockCode",
        "rows",
        "units",
        "line_value",
        "example_description",
    ]
    assert table["reason_code"].tolist() == [
        "listed_non_product_code",
        "listed_non_product_code",
        "listed_unidentifiable_code",
    ]
    assert table["StockCode"].tolist() == ["POST", "M", "DCGSSGIRL"]
    assert table["rows"].tolist() == [3, 1, 1]
    assert table["units"].tolist() == pytest.approx([4.0, -1.0, 3.0])
    assert table["line_value"].tolist() == pytest.approx([20.0, -8.0, 6.0])
    assert table["example_description"].tolist() == ["POSTAGE", "Manual", "SYNTHETIC ITEM"]


def test_zero_price_breakdown_by_quantity_sign(report):
    table = report.zero_price_breakdown
    assert list(table.columns) == [
        "quantity_sign",
        "rows",
        "units",
        "line_value",
        "missing_description",
        "without_customer_id",
    ]
    assert table["quantity_sign"].tolist() == ["positive", "negative", "zero"]
    assert table["rows"].tolist() == [2, 1, 1]
    assert table["units"].tolist() == pytest.approx([7.0, -4.0, 0.0])
    assert table["line_value"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert table["missing_description"].tolist() == [2, 0, 0]
    assert table["without_customer_id"].tolist() == [1, 1, 0]
    assert int(table["rows"].sum()) == EXPECTED_EXCLUDED["zero_price"][0]


def test_duplicate_exposure_covers_every_retained_flagged_row_once(annotated, report):
    table = report.duplicate_exposure
    assert list(table.columns) == ["analytic_role", "rows", "units", "line_value"]
    assert table["analytic_role"].tolist() == [
        "merchandise_sale",
        "merchandise_return",
        "voucher",
    ]
    assert table["rows"].tolist() == [1, 1, 1]
    assert table["units"].tolist() == pytest.approx([4.0, -2.0, 1.0])
    assert table["line_value"].tolist() == pytest.approx([10.0, -3.0, 20.0])
    retained_flagged = annotated.loc[
        annotated["is_exact_duplicate"] & ~annotated["is_excluded"], "row_id"
    ].tolist()
    assert retained_flagged == [3, 6, 8]
    assert int(table["rows"].sum()) == len(retained_flagged)
    # The excluded repeat (a repeated POST row) is counted separately, once.
    assert report.excluded_flagged_duplicate_rows == 1
    # Flagged rows stay inside the merchandise totals; nothing is excluded for them.
    assert report.merchandise_sales.rows == 4


# ---------------------------- small scenario tests ----------------------------
def test_overlap_copy_value_is_reported_as_excluded():
    rows = [
        make_row("1", "85123A", 2, 3.0),
        make_row("2", "85123A", 1, 1.0, date=WINDOW),
        make_row("2", "85123A", 1, 1.0, date=WINDOW, sheet="S2"),
    ]
    report = build_revenue_report(annotate(rows))
    assert_totals(report.merchandise_sales, rows=2, invoices=2, units=3.0, value=7.0)
    by_reason = report.excluded_by_reason.set_index("reason_code")
    assert int(by_reason.loc["sheet_overlap_copy", "rows"]) == 1
    assert float(by_reason.loc["sheet_overlap_copy", "units"]) == pytest.approx(1.0)
    assert float(by_reason.loc["sheet_overlap_copy", "line_value"]) == pytest.approx(1.0)
    assert_totals(report.excluded_total, rows=1, invoices=1, units=1.0, value=1.0)
    assert int(report.duplicate_exposure["rows"].sum()) == 0
    assert all(check.passed for check in report.checks)


def test_duplicate_policy_flag_versus_drop():
    rows = [make_row("1", "85123A", 2, 5.0), make_row("1", "85123A", 2, 5.0)]

    kept = build_revenue_report(annotate(rows))
    assert_totals(kept.merchandise_sales, rows=2, invoices=1, units=4.0, value=20.0)
    assert kept.duplicate_exposure["rows"].tolist() == [1, 0, 0]
    assert kept.duplicate_exposure["units"].tolist() == pytest.approx([2.0, 0.0, 0.0])
    assert kept.duplicate_exposure["line_value"].tolist() == pytest.approx([10.0, 0.0, 0.0])
    assert kept.excluded_flagged_duplicate_rows == 0

    dropped = build_revenue_report(annotate(rows, duplicate_policy="drop"))
    assert_totals(dropped.merchandise_sales, rows=1, invoices=1, units=2.0, value=10.0)
    by_reason = dropped.excluded_by_reason.set_index("reason_code")
    assert int(by_reason.loc["exact_duplicate_dropped", "rows"]) == 1
    assert float(by_reason.loc["exact_duplicate_dropped", "units"]) == pytest.approx(2.0)
    assert float(by_reason.loc["exact_duplicate_dropped", "line_value"]) == pytest.approx(10.0)
    assert dropped.duplicate_exposure["rows"].tolist() == [0, 0, 0]
    assert dropped.excluded_flagged_duplicate_rows == 1


def test_cancellation_exclude_policy_moves_value_to_excluded():
    rows = [
        make_row("1", "85123A", 2, 3.0),
        make_row("C2", "85123A", -1, 3.0),
        make_row("C3", "gift_0001_20", -1, 20.0),
    ]
    report = build_revenue_report(annotate(rows, cancellation_policy="exclude"))
    assert_totals(report.merchandise_sales, rows=1, invoices=1, units=2.0, value=6.0)
    assert_totals(report.merchandise_returns, rows=0, invoices=0, units=0.0, value=0.0)
    assert_totals(report.voucher_net, rows=0, invoices=0, units=0.0, value=0.0)
    by_reason = report.excluded_by_reason.set_index("reason_code")
    assert int(by_reason.loc["cancellation_excluded", "rows"]) == 2
    assert float(by_reason.loc["cancellation_excluded", "units"]) == pytest.approx(-2.0)
    assert float(by_reason.loc["cancellation_excluded", "line_value"]) == pytest.approx(-23.0)
    assert all(check.passed for check in report.checks)


def test_input_with_only_excluded_rows():
    report = build_revenue_report(
        annotate([make_row("1", "POST", 1, 5.0, description="POSTAGE")])
    )
    for totals in (
        report.merchandise_sales,
        report.merchandise_returns,
        report.net_merchandise,
        report.voucher_sales,
        report.voucher_returns,
        report.voucher_net,
    ):
        assert_totals(totals, rows=0, invoices=0, units=0.0, value=0.0)
    assert_totals(report.excluded_total, rows=1, invoices=1, units=1.0, value=5.0)
    assert report.listed_code_values["StockCode"].tolist() == ["POST"]
    assert report.zero_price_breakdown["rows"].tolist() == [0, 0, 0]
    assert report.duplicate_exposure["rows"].tolist() == [0, 0, 0]
    assert all(check.passed for check in report.checks)


# --------------------------------- empty input --------------------------------
def test_empty_input_produces_an_all_zero_report():
    report = build_revenue_report(annotate([]))
    assert report.total_rows == 0
    for totals in (
        report.merchandise_sales,
        report.merchandise_returns,
        report.net_merchandise,
        report.voucher_sales,
        report.voucher_returns,
        report.voucher_net,
        report.excluded_total,
    ):
        assert_totals(totals, rows=0, invoices=0, units=0.0, value=0.0)
    assert report.excluded_by_reason["reason_code"].tolist() == list(EXPECTED_EXCLUDED)
    assert int(report.excluded_by_reason["rows"].sum()) == 0
    assert report.listed_code_values.empty
    assert list(report.listed_code_values.columns) == [
        "reason_code",
        "StockCode",
        "rows",
        "units",
        "line_value",
        "example_description",
    ]
    assert report.zero_price_breakdown["quantity_sign"].tolist() == [
        "positive",
        "negative",
        "zero",
    ]
    assert report.zero_price_breakdown["rows"].tolist() == [0, 0, 0]
    assert report.duplicate_exposure["rows"].tolist() == [0, 0, 0]
    assert report.excluded_flagged_duplicate_rows == 0
    assert len(report.checks) == 12
    assert all(check.passed for check in report.checks)


# ------------------------- tampering with a finished report --------------------
def _change_totals(report, field, **changes):
    return replace(report, **{field: replace(getattr(report, field), **changes)})


def _change_table(report, table_name, locate, column, delta):
    table = getattr(report, table_name).copy()
    label = locate(table)
    table.loc[label, column] = table.loc[label, column] + delta
    return replace(report, **{table_name: table})


def _reason_label(code):
    return lambda table: table.index[(table["reason_code"] == code).to_numpy()][0]


def _first_label(table):
    return table.index[0]


def _sign_label(sign):
    return lambda table: table.index[(table["quantity_sign"] == sign).to_numpy()][0]


def _role_label(role):
    return lambda table: table.index[(table["analytic_role"] == role).to_numpy()][0]


TAMPERS = [
    pytest.param(
        lambda r: _change_totals(r, "merchandise_sales", rows=r.merchandise_sales.rows + 1),
        "every annotated row is in exactly one section",
        id="sales_rows_plus_one",
    ),
    pytest.param(
        lambda r: replace(r, total_rows=r.total_rows + 1),
        "every annotated row is in exactly one section",
        id="total_rows_changed",
    ),
    pytest.param(
        lambda r: _change_totals(r, "net_merchandise", value=r.net_merchandise.value + 1.0),
        "value: net merchandise",
        id="net_value_plus_one",
    ),
    pytest.param(
        lambda r: _change_totals(r, "net_merchandise", units=r.net_merchandise.units + 1.0),
        "units: net merchandise",
        id="net_units_plus_one",
    ),
    pytest.param(
        lambda r: _change_totals(r, "merchandise_sales", value=r.merchandise_sales.value + 1.0),
        "net merchandise = sales",
        id="sales_value_only",
    ),
    pytest.param(
        lambda r: _change_totals(r, "voucher_net", value=r.voucher_net.value + 1.0),
        "net vouchers = voucher sales",
        id="voucher_net_value",
    ),
    pytest.param(
        lambda r: _change_totals(r, "excluded_total", value=r.excluded_total.value + 1.0),
        "excluded-by-reason table adds up",
        id="excluded_total_value",
    ),
    pytest.param(
        lambda r: _change_table(
            r, "excluded_by_reason", _reason_label("zero_price"), "rows", 1
        ),
        "excluded-by-reason table adds up",
        id="excluded_table_zero_price_rows",
    ),
    pytest.param(
        lambda r: _change_table(
            r, "excluded_by_reason", _reason_label("invalid_field"), "rows_without_value", 1
        ),
        "rows without a computable value",
        id="excluded_table_missing_values",
    ),
    pytest.param(
        lambda r: _change_table(r, "listed_code_values", _first_label, "line_value", 1.0),
        "per-code table equals",
        id="listed_code_value",
    ),
    pytest.param(
        lambda r: _change_table(
            r, "zero_price_breakdown", _sign_label("positive"), "units", 1.0
        ),
        "zero-price breakdown equals",
        id="zero_price_units",
    ),
    pytest.param(
        lambda r: _change_table(
            r, "duplicate_exposure", _role_label("merchandise_sale"), "rows", 1
        ),
        "duplicate exposure covers",
        id="duplicate_exposure_rows",
    ),
    pytest.param(
        lambda r: replace(
            r, excluded_flagged_duplicate_rows=r.excluded_flagged_duplicate_rows + 1
        ),
        "duplicate exposure covers",
        id="excluded_flagged_count",
    ),
]


@pytest.mark.parametrize("tamper,fragment", TAMPERS)
def test_tampered_report_is_rejected(annotated, report, tamper, fragment):
    tampered = tamper(report)
    with pytest.raises(RevenueReconciliationError) as excinfo:
        verify_revenue_report(tampered, annotated)
    assert fragment in str(excinfo.value)
    # The untouched report still verifies, so tampering did not change it.
    assert all(check.passed for check in verify_revenue_report(report, annotated))


def test_difference_within_tolerance_is_accepted(annotated, report):
    nudged = _change_totals(
        report, "net_merchandise", value=report.net_merchandise.value + 0.005
    )
    assert all(check.passed for check in verify_revenue_report(nudged, annotated))


def test_difference_beyond_tolerance_is_rejected(annotated, report):
    nudged = _change_totals(
        report, "net_merchandise", value=report.net_merchandise.value + 0.02
    )
    with pytest.raises(RevenueReconciliationError, match="value: net merchandise"):
        verify_revenue_report(nudged, annotated)


def test_wider_tolerance_accepts_a_larger_difference(annotated, report):
    nudged = _change_totals(
        report, "net_merchandise", value=report.net_merchandise.value + 0.02
    )
    checks = verify_revenue_report(nudged, annotated, tolerance=0.05)
    assert all(check.passed for check in checks)


def test_revenue_errors_are_separate_from_cleaning_errors():
    assert issubclass(RevenueReconciliationError, RevenueError)
    assert not issubclass(RevenueReconciliationError, ReconciliationError)
    assert not issubclass(ReconciliationError, RevenueError)


# ----------------------- invalid annotated input (cleaning errors) -------------
def _duplicate_row_id(d):
    d.loc[1, "row_id"] = 0


def _wrong_disposition(d):
    d.loc[0, "disposition"] = "excluded"


def _wrong_role(d):
    d.loc[0, "analytic_role"] = "voucher"


def _extra_voucher_flag(d):
    d.loc[0, "is_voucher"] = True


def _wrong_basket_flag(d):
    d.loc[4, "is_basket_eligible"] = True


def _unknown_reason(d):
    d.loc[0, "reason_code"] = "mystery"


@pytest.mark.parametrize(
    "mutate,message",
    [
        pytest.param(_duplicate_row_id, "row_id", id="duplicate_row_id"),
        pytest.param(_wrong_disposition, "disposition does not match", id="wrong_disposition"),
        pytest.param(_wrong_role, "analytic_role does not match", id="wrong_role"),
        pytest.param(_extra_voucher_flag, "is_voucher does not match", id="extra_voucher_flag"),
        pytest.param(_wrong_basket_flag, "is_basket_eligible", id="wrong_basket_flag"),
        pytest.param(_unknown_reason, "Unknown", id="unknown_reason"),
    ],
)
def test_invalid_annotated_input_raises_the_cleaning_error(annotated, mutate, message):
    tampered = annotated.copy()
    mutate(tampered)
    with pytest.raises(ReconciliationError, match=message) as excinfo:
        build_revenue_report(tampered)
    assert not isinstance(excinfo.value, RevenueReconciliationError)


def test_missing_added_column_raises_the_cleaning_error(annotated):
    with pytest.raises(ReconciliationError, match="missing column"):
        build_revenue_report(annotated.drop(columns=["row_id"]))


@pytest.mark.parametrize(
    "column", ["Invoice", "reason_code", "line_value", "has_customer_id"]
)
def test_missing_required_column_raises_schema_error(annotated, column):
    with pytest.raises(SchemaError, match=column):
        build_revenue_report(annotated.drop(columns=[column]))


def test_verify_with_a_missing_column_raises_schema_error(annotated, report):
    with pytest.raises(SchemaError, match="quantity_num"):
        verify_revenue_report(report, annotated.drop(columns=["quantity_num"]))


# --------------------------------- text report ---------------------------------
def test_text_contains_all_sections_and_totals(report):
    text = format_revenue_report(report)
    for heading in (
        "REVENUE REPORT",
        "1. MERCHANDISE",
        "2. VOUCHERS",
        "3. EXCLUDED ROWS BY REASON",
        "4. EXCLUDED LISTED CODES",
        "5. ZERO-PRICE ROWS BY QUANTITY SIGN",
        "6. DUPLICATE EXPOSURE",
        "7. RECONCILIATION",
        "NOTES",
    ):
        assert heading in text, heading
    for line in (
        "Annotated rows: 26",
        "Sales: rows 4, invoices 2, units 11.00, value GBP 27.50",
        "Returns: rows 3, invoices 2, units -5.00, value GBP -9.00",
        "Net merchandise (net of recorded returns): rows 7, invoices 4, "
        "units 6.00, value GBP 18.50",
        "Voucher sales: rows 2, invoices 1, units 2.00, value GBP 40.00",
        "Voucher returns: rows 1, invoices 1, units -1.00, value GBP -20.00",
        "Net vouchers: rows 3, invoices 2, units 1.00, value GBP 20.00",
        "Excluded total: rows 16, invoices 15, units 13.00, value GBP 21.50",
        "Flagged rows excluded for another reason: 1",
    ):
        assert line in text, line


def test_text_shows_every_check_and_ends_with_the_notes(report):
    text = format_revenue_report(report)
    assert text.count("[PASS]") == 12
    assert "[FAIL]" not in text
    assert "tolerance GBP 0.01" in text
    assert "NOT profit" in text
    assert "no field linking a return to its original purchase" in text
    last_line = text.rstrip().splitlines()[-1]
    assert last_line.startswith("- Rows are classified by the FIRST matching rule")


def test_text_for_a_report_without_recorded_checks(report):
    text = format_revenue_report(replace(report, checks=()))
    assert "no checks recorded" in text
    assert "[PASS]" not in text


def test_text_for_an_empty_report():
    text = format_revenue_report(build_revenue_report(annotate([])))
    assert "Annotated rows: 0" in text
    assert "(none)" in text
    assert text.count("[PASS]") == 12
    assert "NOTES" in text