"""Annotate every raw transaction row with exactly one disposition and reason.

The raw data is never modified or dropped. The result is an annotated copy with
the same rows as the input, plus new columns. Raw columns keep their original
values; parsed numbers and dates go into NEW columns.

Precedence: reasons are listed in src.preprocessing.rules.REASONS; the FIRST
matching reason wins, so every row gets exactly one reason.

Local-only: no file access, no network calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import CleaningConfig
from src.ingestion.loader import SOURCE_COLUMN, validate_schema
from src.preprocessing import rules
from src.validation.overlap import find_mismatched_invoices
from src.validation.quality import STANDARD_STOCK_CODE_PATTERN

logger = logging.getLogger(__name__)

CORE_COLUMNS: tuple[str, ...] = (
    "Invoice",
    "StockCode",
    "InvoiceDate",
    "Quantity",
    "Price",
    "Customer ID",
)
ADDED_COLUMNS: tuple[str, ...] = (
    "row_id",
    "invoice_datetime",
    "quantity_num",
    "price_num",
    "line_value",
    "has_customer_id",
    "is_cancellation_invoice",
    "is_voucher_code",
    "is_exact_duplicate",
    "reason_code",
    "disposition",
    "analytic_role",
    "is_merchandise_sale",
    "is_merchandise_return",
    "is_voucher",
    "is_excluded",
    "is_basket_eligible",
)
ROLE_FLAG_COLUMNS: tuple[str, ...] = (
    "is_merchandise_sale",
    "is_merchandise_return",
    "is_voucher",
    "is_excluded",
)
VALUE_TOLERANCE: float = 0.01
MAX_MISMATCH_EXAMPLES: int = 20


class CleaningError(Exception):
    """Raised when cleaning cannot proceed safely."""


class OverlapMismatchError(CleaningError):
    """Raised when invoices shared by two sheets are not identical."""

    def __init__(self, first_sheet: str, second_sheet: str, mismatched: list[str]):
        self.first_sheet = first_sheet
        self.second_sheet = second_sheet
        self.mismatched_count = len(mismatched)
        self.examples = tuple(mismatched[:MAX_MISMATCH_EXAMPLES])
        super().__init__(
            f"{len(mismatched)} invoice(s) present in both '{first_sheet}' and "
            f"'{second_sheet}' have different rows. Nothing was removed. "
            f"Examples: {list(self.examples)}"
        )


class ReconciliationError(CleaningError):
    """Raised when the row accounting checks fail."""


@dataclass(frozen=True)
class CleaningResult:
    annotated: pd.DataFrame
    reason_summary: pd.DataFrame
    sheets_in_order: tuple[str, ...]
    overlap_invoices_in_both: int
    overlap_rows_excluded: int


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _bool_array(series: pd.Series) -> np.ndarray:
    """Convert a boolean-like Series (possibly with missing values) to numpy bool.

    Returns a writable COPY: under pandas copy-on-write, to_numpy can return a
    read-only view, which breaks in-place operations such as |=.
    """
    return series.fillna(False).to_numpy(dtype=bool, copy=True)


def _parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _lookup_from_reasons(
    reason_values: np.ndarray, error: type[CleaningError] = CleaningError
) -> tuple[np.ndarray, np.ndarray]:
    """Map reason codes to (disposition, analytic role) arrays."""
    codes = pd.Index(list(rules.REASON_CODES)).get_indexer(reason_values)
    if (codes < 0).any():
        raise error("Unknown or missing reason code encountered.")
    dispositions = np.array([i.disposition for i in rules.REASONS], dtype=object)
    roles = np.array([i.role for i in rules.REASONS], dtype=object)
    return dispositions[codes], roles[codes]


def _order_sheets(out: pd.DataFrame, invoice_datetime: pd.Series) -> list[str]:
    """Sheets ordered by earliest valid invoice date (ties: order of appearance)."""
    sheets = [str(s) for s in out[SOURCE_COLUMN].drop_duplicates().tolist()]
    sort_keys: dict[str, tuple[pd.Timestamp, int]] = {}
    for position, sheet in enumerate(sheets):
        mask = _bool_array(out[SOURCE_COLUMN] == sheet)
        first_date = invoice_datetime[mask].min()
        sort_keys[sheet] = (
            first_date if pd.notna(first_date) else pd.Timestamp.max,
            position,
        )
    return sorted(sheets, key=lambda name: sort_keys[name])


def _overlap_copy_mask(
    out: pd.DataFrame,
    invoice_datetime: pd.Series,
    policy: str,
    keys: Sequence[str],
) -> tuple[np.ndarray, list[str], int]:
    """Find later-sheet copies of invoices; fail safely if content differs."""
    none = np.zeros(len(out), dtype=bool)
    sheets = _order_sheets(out, invoice_datetime)
    if policy == "keep_all" or len(sheets) < 2:
        return none, sheets, 0
    if len(sheets) > 2:
        raise CleaningError(
            "overlap_policy 'keep_earlier_sheet' supports at most two sheets, "
            f"found {len(sheets)}. Choose sheet_names explicitly or use 'keep_all'."
        )
    first, second = sheets
    shared, mismatched = find_mismatched_invoices(out, first, second, keys)
    if mismatched:
        raise OverlapMismatchError(first, second, sorted(mismatched))
    in_second = _bool_array(out[SOURCE_COLUMN] == second)
    in_shared = out["Invoice"].isin(shared).to_numpy(dtype=bool)
    return in_second & in_shared, sheets, len(shared)


# ----------------------------------------------------------------------------
# Main cleaning function
# ----------------------------------------------------------------------------
def clean_transactions(
    raw: pd.DataFrame,
    cleaning: CleaningConfig,
    required_columns: Sequence[str],
) -> CleaningResult:
    """Return an annotated copy of `raw` with one reason per row."""
    keys = list(dict.fromkeys(required_columns))
    validate_schema(
        list(raw.columns), [*CORE_COLUMNS, *keys, SOURCE_COLUMN], "cleaning input"
    )
    clashes = [name for name in ADDED_COLUMNS if name in raw.columns]
    if clashes:
        raise CleaningError(f"Input already contains reserved column(s): {clashes}")
    if raw[SOURCE_COLUMN].isna().any():
        raise CleaningError(f"'{SOURCE_COLUMN}' must not contain missing values.")

    raw_columns = list(raw.columns)
    n_rows = len(raw)
    out = raw.reset_index(drop=True)

    # --- text fields (raw values are matched exactly; nothing is rewritten) ---
    invoice = out["Invoice"].astype("string")
    code = out["StockCode"].astype("string")
    invoice_ok = _bool_array(invoice.notna() & (invoice.str.strip() != ""))
    code_ok = _bool_array(code.notna() & (code.str.strip() != ""))

    is_cancel = _bool_array(invoice.str.startswith(cleaning.cancellation_invoice_prefix))
    starts_with_letter = _bool_array(invoice.str.match(r"^[A-Za-z]"))
    bad_prefix = starts_with_letter & ~is_cancel

    is_voucher = np.zeros(n_rows, dtype=bool)
    for prefix in cleaning.voucher_stock_code_prefixes:
        is_voucher |= _bool_array(code.str.startswith(prefix))
    listed_non_product = _bool_array(code.isin(list(cleaning.non_product_stock_codes)))
    listed_unidentifiable = _bool_array(
        code.isin(list(cleaning.unidentifiable_stock_codes))
    )

    # --- numeric and date fields go into NEW columns ---
    quantity = _parse_numeric(out["Quantity"])
    price = _parse_numeric(out["Price"])
    line_value = quantity * price
    invoice_datetime = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    qty = quantity.to_numpy()
    prc = price.to_numpy()
    date_ok = invoice_datetime.notna().to_numpy(dtype=bool)

    invalid = ~(
        invoice_ok & code_ok & date_ok & np.isfinite(qty) & np.isfinite(prc)
    )

    # --- overlap between sheets (fails safely if content differs) ---
    overlap_copy, sheets, shared_count = _overlap_copy_mask(
        out, invoice_datetime, cleaning.overlap_policy, keys
    )

    # --- exact duplicates among rows that are not overlap copies ---
    is_duplicate = np.zeros(n_rows, dtype=bool)
    candidates = ~overlap_copy
    if candidates.any():
        repeated = out.loc[candidates, keys].duplicated(keep="first")
        is_duplicate[candidates] = repeated.to_numpy(dtype=bool)

    # --- first-match-wins reason assignment ---
    with np.errstate(invalid="ignore"):
        ordered: list[tuple[str, np.ndarray]] = [
            (rules.INVALID_FIELD, invalid),
            (rules.SHEET_OVERLAP_COPY, overlap_copy),
            (
                rules.EXACT_DUPLICATE_DROPPED,
                is_duplicate & (cleaning.duplicate_policy == "drop"),
            ),
            (rules.LISTED_NON_PRODUCT_CODE, listed_non_product),
            (rules.LISTED_UNIDENTIFIABLE_CODE, listed_unidentifiable),
            (rules.UNRECOGNIZED_INVOICE_PREFIX, bad_prefix),
            (rules.NEGATIVE_PRICE, prc < 0),
            (rules.ZERO_PRICE, (prc == 0) & (cleaning.zero_price_policy == "exclude")),
            (
                rules.CANCELLATION_EXCLUDED,
                is_cancel & (cleaning.cancellation_policy == "exclude"),
            ),
            (rules.CANCELLATION_NONNEGATIVE_QUANTITY, is_cancel & (qty >= 0)),
            (rules.NONPOSITIVE_QUANTITY, ~is_cancel & (qty <= 0)),
            (rules.VOUCHER_RETURN, is_voucher & is_cancel),
            (rules.VOUCHER_SALE, is_voucher & ~is_cancel),
            (rules.MERCHANDISE_RETURN, is_cancel),
        ]
    expected_order = list(rules.REASON_CODES[:-1])
    if [name for name, _ in ordered] != expected_order:
        raise CleaningError("Internal error: rule order does not match rules.REASONS.")

    reason = np.full(n_rows, rules.MERCHANDISE_SALE, dtype=object)
    for reason_code, mask in reversed(ordered):  # earlier rules overwrite later ones
        reason[np.asarray(mask, dtype=bool)] = reason_code

    disposition, role = _lookup_from_reasons(reason)

    # --- add the new columns (raw columns are untouched) ---
    out.insert(0, "row_id", np.arange(n_rows, dtype=np.int64))
    out["invoice_datetime"] = invoice_datetime
    out["quantity_num"] = quantity
    out["price_num"] = price
    out["line_value"] = line_value
    out["has_customer_id"] = out["Customer ID"].notna().to_numpy(dtype=bool)
    out["is_cancellation_invoice"] = is_cancel
    out["is_voucher_code"] = is_voucher
    out["is_exact_duplicate"] = is_duplicate
    out["reason_code"] = pd.array(reason, dtype="string")
    out["disposition"] = pd.array(disposition, dtype="string")
    out["analytic_role"] = pd.array(role, dtype="string")
    out["is_merchandise_sale"] = role == rules.ROLE_MERCHANDISE_SALE
    out["is_merchandise_return"] = role == rules.ROLE_MERCHANDISE_RETURN
    out["is_voucher"] = role == rules.ROLE_VOUCHER
    out["is_excluded"] = role == rules.ROLE_EXCLUDED
    out["is_basket_eligible"] = role == rules.ROLE_MERCHANDISE_SALE

    # --- safety checks ---
    if not out[raw_columns].equals(raw.reset_index(drop=True)):
        raise CleaningError("Internal error: raw columns were modified.")
    check_reconciliation(out, n_rows)
    summary = summarize_reasons(out)

    logger.info(
        "Cleaned %d rows: %d retained, %d excluded.",
        n_rows,
        int((~out["is_excluded"]).sum()),
        int(out["is_excluded"].sum()),
    )
    return CleaningResult(
        annotated=out,
        reason_summary=summary,
        sheets_in_order=tuple(sheets),
        overlap_invoices_in_both=shared_count,
        overlap_rows_excluded=int((reason == rules.SHEET_OVERLAP_COPY).sum()),
    )


# ----------------------------------------------------------------------------
# Reconciliation and summaries
# ----------------------------------------------------------------------------
def summarize_reasons(annotated: pd.DataFrame) -> pd.DataFrame:
    """Rows, units and signed line value per reason, in precedence order."""
    validate_schema(
        list(annotated.columns),
        ["reason_code", "quantity_num", "line_value"],
        "reason summary input",
    )
    n_rows = len(annotated)
    values = pd.DataFrame(
        {
            "rows": np.ones(n_rows, dtype=np.int64),
            "units": annotated["quantity_num"].fillna(0.0).to_numpy(dtype=float),
            "line_value": annotated["line_value"].fillna(0.0).to_numpy(dtype=float),
            "rows_without_value": annotated["line_value"]
            .isna()
            .to_numpy(dtype=np.int64),
        }
    )
    keys = annotated["reason_code"].astype(object).to_numpy()
    sums = values.groupby(keys).sum()
    sums = sums.reindex(list(rules.REASON_CODES), fill_value=0)
    sums["rows"] = sums["rows"].astype("int64")
    sums["rows_without_value"] = sums["rows_without_value"].astype("int64")
    sums["units"] = sums["units"].astype("float64")
    sums["line_value"] = sums["line_value"].astype("float64")
    sums.insert(0, "role", [info.role for info in rules.REASONS])
    sums.insert(0, "disposition", [info.disposition for info in rules.REASONS])
    sums.insert(0, "reason_code", list(rules.REASON_CODES))
    return sums.reset_index(drop=True)


def check_reconciliation(
    annotated: pd.DataFrame,
    raw_row_count: int,
    tolerance: float = VALUE_TOLERANCE,
) -> dict[str, int]:
    """Verify that every raw row has exactly one consistent outcome."""
    missing = [name for name in ADDED_COLUMNS if name not in annotated.columns]
    if missing:
        raise ReconciliationError(f"Annotated table is missing column(s): {missing}")

    problems: list[str] = []
    n_rows = len(annotated)
    if n_rows != raw_row_count:
        problems.append(f"row count {n_rows} != raw row count {raw_row_count}")

    if not np.array_equal(annotated["row_id"].to_numpy(), np.arange(n_rows)):
        problems.append("row_id is not 0..n-1 without gaps or repeats")

    reason_values = annotated["reason_code"].astype(object).to_numpy()
    disposition, role = _lookup_from_reasons(reason_values, ReconciliationError)

    if not np.array_equal(
        annotated["disposition"].astype(object).to_numpy(), disposition
    ):
        problems.append("disposition does not match reason_code")
    if not np.array_equal(annotated["analytic_role"].astype(object).to_numpy(), role):
        problems.append("analytic_role does not match reason_code")

    expected_flags = {
        "is_merchandise_sale": role == rules.ROLE_MERCHANDISE_SALE,
        "is_merchandise_return": role == rules.ROLE_MERCHANDISE_RETURN,
        "is_voucher": role == rules.ROLE_VOUCHER,
        "is_excluded": role == rules.ROLE_EXCLUDED,
    }
    for flag, expected in expected_flags.items():
        if not np.array_equal(annotated[flag].to_numpy(dtype=bool), expected):
            problems.append(f"{flag} does not match analytic_role")

    flag_matrix = np.column_stack(
        [annotated[flag].to_numpy(dtype=bool) for flag in ROLE_FLAG_COLUMNS]
    ) if n_rows else np.zeros((0, len(ROLE_FLAG_COLUMNS)), dtype=bool)
    if not (flag_matrix.sum(axis=1) == 1).all():
        problems.append("not every row has exactly one role flag")

    if not np.array_equal(
        annotated["is_basket_eligible"].to_numpy(dtype=bool),
        annotated["is_merchandise_sale"].to_numpy(dtype=bool),
    ):
        problems.append("is_basket_eligible differs from is_merchandise_sale")

    retained = int((disposition == rules.RETAINED).sum())
    excluded = int((disposition == rules.EXCLUDED).sum())
    if retained + excluded != n_rows:
        problems.append("retained + excluded != total rows")

    summary = summarize_reasons(annotated)
    if int(summary["rows"].sum()) != n_rows:
        problems.append("per-reason row counts do not add up to total rows")
    total_value = float(annotated["line_value"].sum())
    if abs(float(summary["line_value"].sum()) - total_value) > tolerance:
        problems.append("per-reason values do not add up to total line value")

    if problems:
        raise ReconciliationError("; ".join(problems))
    return {"total_rows": n_rows, "retained_rows": retained, "excluded_rows": excluded}


# ----------------------------------------------------------------------------
# Selectors (filter by flags only)
# ----------------------------------------------------------------------------
def _select(annotated: pd.DataFrame, flag: str) -> pd.DataFrame:
    validate_schema(list(annotated.columns), [flag], "selector input")
    return annotated.loc[annotated[flag].to_numpy(dtype=bool)]


def select_merchandise_sales(annotated: pd.DataFrame) -> pd.DataFrame:
    return _select(annotated, "is_merchandise_sale")


def select_merchandise_returns(annotated: pd.DataFrame) -> pd.DataFrame:
    return _select(annotated, "is_merchandise_return")


def select_vouchers(annotated: pd.DataFrame) -> pd.DataFrame:
    return _select(annotated, "is_voucher")


def select_excluded(annotated: pd.DataFrame) -> pd.DataFrame:
    return _select(annotated, "is_excluded")


def select_basket_eligible(annotated: pd.DataFrame) -> pd.DataFrame:
    return _select(annotated, "is_basket_eligible")


# ----------------------------------------------------------------------------
# Audit and display helpers
# ----------------------------------------------------------------------------
def audit_uncovered_unusual_codes(
    annotated: pd.DataFrame, cleaning: CleaningConfig
) -> pd.DataFrame:
    """List non-standard stock codes that no configured list covers.

    These codes are NOT excluded because of their format; this table is the
    audit trail so they can be reviewed.
    """
    validate_schema(
        list(annotated.columns),
        ["StockCode", "Description", "is_excluded"],
        "unusual code audit input",
    )
    codes = annotated["StockCode"].astype("string")
    standard = _bool_array(codes.str.match(STANDARD_STOCK_CODE_PATTERN))
    listed = list(cleaning.non_product_stock_codes) + list(
        cleaning.unidentifiable_stock_codes
    )
    covered = _bool_array(codes.isin(listed))
    for prefix in cleaning.voucher_stock_code_prefixes:
        covered |= _bool_array(codes.str.startswith(prefix))
    mask = ~standard & ~covered & _bool_array(codes.notna())

    columns = ["StockCode", "rows", "retained_rows", "example_description"]
    if not mask.any():
        return pd.DataFrame(columns=columns)
    subset = annotated.loc[mask, ["StockCode", "Description", "is_excluded"]].assign(
        retained=lambda frame: ~frame["is_excluded"].astype(bool)
    )
    result = (
        subset.groupby("StockCode")
        .agg(
            rows=("retained", "size"),
            retained_rows=("retained", "sum"),
            example_description=("Description", "first"),
        )
        .reset_index()
        .sort_values(["rows", "StockCode"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return result[columns]


def format_reason_summary(summary: pd.DataFrame) -> str:
    """Plain-text table of the per-reason summary, with totals."""
    table = summary.to_string(
        index=False, float_format=lambda value: f"{value:,.2f}"
    )
    retained = int(summary.loc[summary["disposition"] == rules.RETAINED, "rows"].sum())
    excluded = int(summary.loc[summary["disposition"] == rules.EXCLUDED, "rows"].sum())
    return (
        f"{table}\n\nTotal rows: {retained + excluded} "
        f"(retained {retained}, excluded {excluded})"
    )