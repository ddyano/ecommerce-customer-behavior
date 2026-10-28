"""Revenue report built from the annotated cleaning output.

All values are sterling LINE VALUE (Quantity x Price). They are NOT profit
(there is no cost data). The dataset has no field linking a return to its
original purchase.

The report shows, separately: merchandise sales, merchandise returns, net
merchandise, vouchers, excluded values by reason, per-code values for excluded
listed codes, a zero-price breakdown by quantity sign, and the value of
retained rows flagged as exact duplicates (which are NOT excluded here).
Reconciliation checks (tolerance GBP 0.01 on value and units) raise an error
instead of ever displaying a report that does not add up.

Local-only: no file access, no network calls.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.ingestion.loader import validate_schema
from src.preprocessing import rules
from src.preprocessing.cleaning import check_reconciliation, summarize_reasons

REVENUE_TOLERANCE: float = 0.01

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Invoice",
    "StockCode",
    "Description",
    "quantity_num",
    "line_value",
    "has_customer_id",
    "is_exact_duplicate",
    "reason_code",
    "analytic_role",
    "is_merchandise_sale",
    "is_merchandise_return",
    "is_voucher",
    "is_excluded",
)
LISTED_REASONS: tuple[str, ...] = (
    rules.LISTED_NON_PRODUCT_CODE,
    rules.LISTED_UNIDENTIFIABLE_CODE,
)
DUPLICATE_ROLES: tuple[tuple[str, str], ...] = (
    (rules.ROLE_MERCHANDISE_SALE, "is_merchandise_sale"),
    (rules.ROLE_MERCHANDISE_RETURN, "is_merchandise_return"),
    (rules.ROLE_VOUCHER, "is_voucher"),
)
SUMMARY_COLUMNS: list[str] = [
    "reason_code",
    "rows",
    "units",
    "line_value",
    "rows_without_value",
]
LISTED_COLUMNS: list[str] = [
    "reason_code",
    "StockCode",
    "rows",
    "units",
    "line_value",
    "example_description",
]
ZERO_PRICE_COLUMNS: list[str] = [
    "quantity_sign",
    "rows",
    "units",
    "line_value",
    "missing_description",
    "without_customer_id",
]
DUPLICATE_COLUMNS: list[str] = ["analytic_role", "rows", "units", "line_value"]

NOTES: tuple[str, ...] = (
    "All values are sterling line value (Quantity x Price). This is NOT profit: "
    "the dataset has no cost data.",
    "The dataset has no field linking a return to its original purchase, so one "
    "period's returns can relate to sales from earlier periods.",
    "Excluded rows are not part of any merchandise or voucher total; their values "
    "are shown separately above.",
    "Merchandise totals INCLUDE retained rows flagged as exact duplicates "
    "(duplicate_policy 'flag'); their value is reported separately in the "
    "duplicate-exposure section.",
    "Rows are classified by the FIRST matching rule, so a row that fits several "
    "rules is counted only under the first.",
)


class RevenueError(Exception):
    """Base class for revenue report errors."""


class RevenueReconciliationError(RevenueError):
    """Raised when the revenue report does not reconcile."""


@dataclass(frozen=True)
class Totals:
    rows: int
    invoices: int
    units: float
    value: float


@dataclass(frozen=True)
class ReportCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, eq=False)
class RevenueReport:
    total_rows: int
    tolerance: float
    merchandise_sales: Totals
    merchandise_returns: Totals
    net_merchandise: Totals
    voucher_sales: Totals
    voucher_returns: Totals
    voucher_net: Totals
    excluded_total: Totals
    excluded_by_reason: pd.DataFrame
    listed_code_values: pd.DataFrame
    zero_price_breakdown: pd.DataFrame
    duplicate_exposure: pd.DataFrame
    excluded_flagged_duplicate_rows: int
    checks: tuple[ReportCheck, ...] = ()


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def _flag(annotated: pd.DataFrame, column: str) -> np.ndarray:
    return annotated[column].to_numpy(dtype=bool)


def _reason_mask(annotated: pd.DataFrame, *codes: str) -> np.ndarray:
    return annotated["reason_code"].isin(list(codes)).to_numpy(dtype=bool)


def _totals(subset: pd.DataFrame) -> Totals:
    return Totals(
        rows=int(len(subset)),
        invoices=int(subset["Invoice"].nunique()),
        units=float(subset["quantity_num"].sum()),
        value=float(subset["line_value"].sum()),
    )


def _listed_code_values(annotated: pd.DataFrame) -> pd.DataFrame:
    mask = _reason_mask(annotated, *LISTED_REASONS)
    subset = annotated.loc[
        mask, ["reason_code", "StockCode", "Description", "quantity_num", "line_value"]
    ]
    if subset.empty:
        return pd.DataFrame(columns=LISTED_COLUMNS)
    grouped = (
        subset.groupby(["reason_code", "StockCode"], sort=False)
        .agg(
            rows=("line_value", "size"),
            units=("quantity_num", "sum"),
            line_value=("line_value", "sum"),
            example_description=("Description", "first"),
        )
        .reset_index()
    )
    order = {code: position for position, code in enumerate(rules.REASON_CODES)}
    grouped["_order"] = grouped["reason_code"].astype(object).map(order)
    grouped = grouped.sort_values(
        ["_order", "rows", "StockCode"], ascending=[True, False, True]
    )
    return grouped.drop(columns="_order").reset_index(drop=True)[LISTED_COLUMNS]


def _zero_price_breakdown(annotated: pd.DataFrame) -> pd.DataFrame:
    subset = annotated.loc[_reason_mask(annotated, rules.ZERO_PRICE)]
    quantity = subset["quantity_num"].to_numpy(dtype=float)
    value = subset["line_value"].to_numpy(dtype=float)
    missing_description = subset["Description"].isna().to_numpy(dtype=bool)
    without_customer = ~subset["has_customer_id"].to_numpy(dtype=bool)
    records = []
    for label, mask in (
        ("positive", quantity > 0),
        ("negative", quantity < 0),
        ("zero", quantity == 0),
    ):
        records.append(
            {
                "quantity_sign": label,
                "rows": int(mask.sum()),
                "units": float(np.nansum(quantity[mask])),
                "line_value": float(np.nansum(value[mask])),
                "missing_description": int((mask & missing_description).sum()),
                "without_customer_id": int((mask & without_customer).sum()),
            }
        )
    return pd.DataFrame(records, columns=ZERO_PRICE_COLUMNS)


def _duplicate_exposure(annotated: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    flagged = _flag(annotated, "is_exact_duplicate")
    excluded = _flag(annotated, "is_excluded")
    retained_flagged = flagged & ~excluded
    records = []
    for role, flag_column in DUPLICATE_ROLES:
        mask = retained_flagged & _flag(annotated, flag_column)
        subset = annotated.loc[mask]
        records.append(
            {
                "analytic_role": role,
                "rows": int(len(subset)),
                "units": float(subset["quantity_num"].sum()),
                "line_value": float(subset["line_value"].sum()),
            }
        )
    table = pd.DataFrame(records, columns=DUPLICATE_COLUMNS)
    return table, int((flagged & excluded).sum())


# ----------------------------------------------------------------------------
# Build and verify
# ----------------------------------------------------------------------------
def build_revenue_report(
    annotated: pd.DataFrame, tolerance: float = REVENUE_TOLERANCE
) -> RevenueReport:
    """Calculate the revenue report and verify that it reconciles."""
    validate_schema(list(annotated.columns), REQUIRED_COLUMNS, "revenue report input")
    check_reconciliation(annotated, len(annotated), tolerance)

    is_sale = _flag(annotated, "is_merchandise_sale")
    is_return = _flag(annotated, "is_merchandise_return")
    is_voucher = _flag(annotated, "is_voucher")
    is_excluded = _flag(annotated, "is_excluded")
    voucher_sale = _reason_mask(annotated, rules.VOUCHER_SALE)
    voucher_return = _reason_mask(annotated, rules.VOUCHER_RETURN)

    summary = summarize_reasons(annotated)
    excluded_table = (
        summary.loc[summary["disposition"] == rules.EXCLUDED, SUMMARY_COLUMNS]
        .reset_index(drop=True)
    )
    duplicate_table, excluded_flagged = _duplicate_exposure(annotated)

    report = RevenueReport(
        total_rows=int(len(annotated)),
        tolerance=float(tolerance),
        merchandise_sales=_totals(annotated.loc[is_sale]),
        merchandise_returns=_totals(annotated.loc[is_return]),
        net_merchandise=_totals(annotated.loc[is_sale | is_return]),
        voucher_sales=_totals(annotated.loc[voucher_sale]),
        voucher_returns=_totals(annotated.loc[voucher_return]),
        voucher_net=_totals(annotated.loc[is_voucher]),
        excluded_total=_totals(annotated.loc[is_excluded]),
        excluded_by_reason=excluded_table,
        listed_code_values=_listed_code_values(annotated),
        zero_price_breakdown=_zero_price_breakdown(annotated),
        duplicate_exposure=duplicate_table,
        excluded_flagged_duplicate_rows=excluded_flagged,
    )
    results = verify_revenue_report(report, annotated, tolerance)
    return replace(report, checks=tuple(results))


def verify_revenue_report(
    report: RevenueReport,
    annotated: pd.DataFrame,
    tolerance: float = REVENUE_TOLERANCE,
) -> list[ReportCheck]:
    """Check a report against the annotated table; raise if anything fails."""
    validate_schema(
        list(annotated.columns), REQUIRED_COLUMNS, "revenue verification input"
    )
    results: list[ReportCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        results.append(ReportCheck(name, bool(passed), detail))

    def close(first: float, second: float) -> bool:
        return abs(float(first) - float(second)) <= tolerance

    n_rows = int(len(annotated))
    sales, returns = report.merchandise_sales, report.merchandise_returns
    v_sales, v_returns = report.voucher_sales, report.voucher_returns
    excluded = report.excluded_total
    net, v_net = report.net_merchandise, report.voucher_net

    counted = sales.rows + returns.rows + v_sales.rows + v_returns.rows + excluded.rows
    add(
        "rows: every annotated row is in exactly one section",
        counted == n_rows and report.total_rows == n_rows,
        f"sections hold {counted}; annotated {n_rows}; report says {report.total_rows}",
    )

    flag_counts = (
        int(_flag(annotated, "is_merchandise_sale").sum()),
        int(_flag(annotated, "is_merchandise_return").sum()),
        int(_flag(annotated, "is_voucher").sum()),
        int(_flag(annotated, "is_excluded").sum()),
    )
    report_counts = (sales.rows, returns.rows, v_sales.rows + v_returns.rows, excluded.rows)
    add(
        "rows: section counts match the role flags",
        flag_counts == report_counts,
        f"flags {flag_counts}; report {report_counts}",
    )

    total_value = float(annotated["line_value"].sum())
    reported_value = net.value + v_net.value + excluded.value
    add(
        f"value: net merchandise + net vouchers + excluded = total (tolerance {tolerance})",
        close(reported_value, total_value),
        f"sections {reported_value:.4f}; total {total_value:.4f}",
    )

    total_units = float(annotated["quantity_num"].sum())
    reported_units = net.units + v_net.units + excluded.units
    add(
        f"units: net merchandise + net vouchers + excluded = total (tolerance {tolerance})",
        close(reported_units, total_units),
        f"sections {reported_units:.4f}; total {total_units:.4f}",
    )

    add(
        "net merchandise = sales + returns",
        net.rows == sales.rows + returns.rows
        and close(net.units, sales.units + returns.units)
        and close(net.value, sales.value + returns.value),
        f"net value {net.value:.4f}; sales + returns {sales.value + returns.value:.4f}",
    )
    add(
        "net vouchers = voucher sales + voucher returns",
        v_net.rows == v_sales.rows + v_returns.rows
        and close(v_net.units, v_sales.units + v_returns.units)
        and close(v_net.value, v_sales.value + v_returns.value),
        f"net value {v_net.value:.4f}; sales + returns {v_sales.value + v_returns.value:.4f}",
    )

    table = report.excluded_by_reason
    add(
        "excluded-by-reason table adds up to the excluded total",
        int(table["rows"].sum()) == excluded.rows
        and close(table["units"].sum(), excluded.units)
        and close(table["line_value"].sum(), excluded.value),
        f"table rows {int(table['rows'].sum())} / excluded rows {excluded.rows}",
    )

    missing_all = int(annotated["line_value"].isna().sum())
    add(
        "rows without a computable value are all in the excluded-by-reason table",
        int(table["rows_without_value"].sum()) == missing_all,
        f"table {int(table['rows_without_value'].sum())}; annotated {missing_all}",
    )

    retained_missing = int(
        (~_flag(annotated, "is_excluded") & annotated["line_value"].isna().to_numpy()).sum()
    )
    add(
        "every retained row has a computable line value",
        retained_missing == 0,
        f"{retained_missing} retained row(s) without a value",
    )

    listed_reasons = table.loc[table["reason_code"].isin(LISTED_REASONS)]
    listed = report.listed_code_values
    add(
        "per-code table equals the listed-code reason rows",
        int(listed["rows"].sum()) == int(listed_reasons["rows"].sum())
        and close(listed["units"].sum(), listed_reasons["units"].sum())
        and close(listed["line_value"].sum(), listed_reasons["line_value"].sum()),
        f"per-code rows {int(listed['rows'].sum())}; reason rows "
        f"{int(listed_reasons['rows'].sum())}",
    )

    zero_reason = table.loc[table["reason_code"] == rules.ZERO_PRICE]
    zero = report.zero_price_breakdown
    add(
        "zero-price breakdown equals the zero_price reason row",
        int(zero["rows"].sum()) == int(zero_reason["rows"].sum())
        and close(zero["units"].sum(), zero_reason["units"].sum())
        and close(zero["line_value"].sum(), zero_reason["line_value"].sum()),
        f"breakdown rows {int(zero['rows'].sum())}; reason rows "
        f"{int(zero_reason['rows'].sum())}",
    )

    flagged = _flag(annotated, "is_exact_duplicate")
    excluded_flag = _flag(annotated, "is_excluded")
    retained_flagged = int((flagged & ~excluded_flag).sum())
    excluded_flagged = int((flagged & excluded_flag).sum())
    add(
        "duplicate exposure covers every retained flagged row",
        int(report.duplicate_exposure["rows"].sum()) == retained_flagged
        and report.excluded_flagged_duplicate_rows == excluded_flagged,
        f"exposure rows {int(report.duplicate_exposure['rows'].sum())}; "
        f"retained flagged {retained_flagged}; excluded flagged "
        f"{report.excluded_flagged_duplicate_rows} (expected {excluded_flagged})",
    )

    failed = [check for check in results if not check.passed]
    if failed:
        raise RevenueReconciliationError(
            "; ".join(f"{check.name}: {check.detail}" for check in failed)
        )
    return results


# ----------------------------------------------------------------------------
# Plain-text report
# ----------------------------------------------------------------------------
def _table_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    return frame.to_string(index=False, float_format=lambda value: f"{value:,.2f}")


def _totals_line(label: str, totals: Totals) -> str:
    return (
        f"{label}: rows {totals.rows:,}, invoices {totals.invoices:,}, "
        f"units {totals.units:,.2f}, value GBP {totals.value:,.2f}"
    )


def format_revenue_report(report: RevenueReport) -> str:
    """Return the report as plain text, always ending with the notes."""
    lines: list[str] = []

    def section(title: str) -> None:
        lines.extend(["", "=" * 78, title, "=" * 78])

    section("REVENUE REPORT (sterling line value, NOT profit)")
    lines.append(f"Annotated rows: {report.total_rows:,}")

    section("1. MERCHANDISE")
    lines += [
        _totals_line("Sales", report.merchandise_sales),
        _totals_line("Returns", report.merchandise_returns),
        _totals_line("Net merchandise (net of recorded returns)", report.net_merchandise),
    ]

    section("2. VOUCHERS (reported separately; never in merchandise totals)")
    lines += [
        _totals_line("Voucher sales", report.voucher_sales),
        _totals_line("Voucher returns", report.voucher_returns),
        _totals_line("Net vouchers", report.voucher_net),
    ]

    section("3. EXCLUDED ROWS BY REASON (not in any total above)")
    lines.append(_table_text(report.excluded_by_reason))
    lines.append(_totals_line("Excluded total", report.excluded_total))

    section("4. EXCLUDED LISTED CODES: VALUE PER CODE")
    lines.append(_table_text(report.listed_code_values))

    section("5. ZERO-PRICE ROWS BY QUANTITY SIGN (final reason zero_price)")
    lines.append(_table_text(report.zero_price_breakdown))

    section("6. DUPLICATE EXPOSURE (retained rows flagged as exact duplicates)")
    lines.append(_table_text(report.duplicate_exposure))
    lines.append(
        "Flagged rows excluded for another reason: "
        f"{report.excluded_flagged_duplicate_rows:,}"
    )

    section(f"7. RECONCILIATION (tolerance GBP {report.tolerance} on value and units)")
    if report.checks:
        for check in report.checks:
            lines.append(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}")
            lines.append(f"       {check.detail}")
    else:
        lines.append("(no checks recorded: report was not built by build_revenue_report)")

    section("NOTES")
    lines += [f"- {note}" for note in NOTES]
    return "\n".join(lines) + "\n"