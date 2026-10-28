"""Reason codes, dispositions and analytical roles for transaction cleaning.

This file is the single source of truth for the cleaning vocabulary.

* disposition: whether a row is RETAINED in the audit output or EXCLUDED.
* analytic role: what a row may be used for (merchandise sale, merchandise
  return, voucher, or nothing because it is excluded).

Reason codes DESCRIBE a condition; they do not interpret what a row "is".
The order of REASONS is the order of precedence: the FIRST matching reason wins.
"""
from __future__ import annotations

from dataclasses import dataclass

RETAINED = "retained"
EXCLUDED = "excluded"

ROLE_MERCHANDISE_SALE = "merchandise_sale"
ROLE_MERCHANDISE_RETURN = "merchandise_return"
ROLE_VOUCHER = "voucher"
ROLE_EXCLUDED = "excluded"
ROLES: tuple[str, ...] = (
    ROLE_MERCHANDISE_SALE,
    ROLE_MERCHANDISE_RETURN,
    ROLE_VOUCHER,
    ROLE_EXCLUDED,
)

INVALID_FIELD = "invalid_field"
SHEET_OVERLAP_COPY = "sheet_overlap_copy"
EXACT_DUPLICATE_DROPPED = "exact_duplicate_dropped"
LISTED_NON_PRODUCT_CODE = "listed_non_product_code"
LISTED_UNIDENTIFIABLE_CODE = "listed_unidentifiable_code"
UNRECOGNIZED_INVOICE_PREFIX = "unrecognized_invoice_prefix"
NEGATIVE_PRICE = "negative_price"
ZERO_PRICE = "zero_price"
CANCELLATION_EXCLUDED = "cancellation_excluded"
CANCELLATION_NONNEGATIVE_QUANTITY = "cancellation_nonnegative_quantity"
NONPOSITIVE_QUANTITY = "nonpositive_quantity"
VOUCHER_RETURN = "voucher_return"
VOUCHER_SALE = "voucher_sale"
MERCHANDISE_RETURN = "merchandise_return"
MERCHANDISE_SALE = "merchandise_sale"


@dataclass(frozen=True)
class ReasonInfo:
    code: str
    disposition: str
    role: str
    description: str


# Precedence order: the first matching reason wins.
REASONS: tuple[ReasonInfo, ...] = (
    ReasonInfo(
        INVALID_FIELD, EXCLUDED, ROLE_EXCLUDED,
        "Invoice, StockCode, InvoiceDate, Quantity or Price is missing or unreadable.",
    ),
    ReasonInfo(
        SHEET_OVERLAP_COPY, EXCLUDED, ROLE_EXCLUDED,
        "Invoice also appears in an earlier sheet with identical rows (later copy).",
    ),
    ReasonInfo(
        EXACT_DUPLICATE_DROPPED, EXCLUDED, ROLE_EXCLUDED,
        "Repeat of an earlier identical row and duplicate_policy is 'drop'.",
    ),
    ReasonInfo(
        LISTED_NON_PRODUCT_CODE, EXCLUDED, ROLE_EXCLUDED,
        "Stock code is in the configured non-product list.",
    ),
    ReasonInfo(
        LISTED_UNIDENTIFIABLE_CODE, EXCLUDED, ROLE_EXCLUDED,
        "Stock code is in the configured unidentifiable list.",
    ),
    ReasonInfo(
        UNRECOGNIZED_INVOICE_PREFIX, EXCLUDED, ROLE_EXCLUDED,
        "Invoice starts with a letter other than the cancellation prefix.",
    ),
    ReasonInfo(NEGATIVE_PRICE, EXCLUDED, ROLE_EXCLUDED, "Price is below zero."),
    ReasonInfo(
        ZERO_PRICE, EXCLUDED, ROLE_EXCLUDED,
        "Price is zero and zero_price_policy is 'exclude'.",
    ),
    ReasonInfo(
        CANCELLATION_EXCLUDED, EXCLUDED, ROLE_EXCLUDED,
        "Cancellation invoice and cancellation_policy is 'exclude'.",
    ),
    ReasonInfo(
        CANCELLATION_NONNEGATIVE_QUANTITY, EXCLUDED, ROLE_EXCLUDED,
        "Cancellation invoice whose quantity is zero or positive.",
    ),
    ReasonInfo(
        NONPOSITIVE_QUANTITY, EXCLUDED, ROLE_EXCLUDED,
        "Non-cancellation invoice whose quantity is zero or negative.",
    ),
    ReasonInfo(
        VOUCHER_RETURN, RETAINED, ROLE_VOUCHER,
        "Voucher stock code on a cancellation invoice (reported separately).",
    ),
    ReasonInfo(
        VOUCHER_SALE, RETAINED, ROLE_VOUCHER,
        "Voucher stock code on a sale invoice (reported separately).",
    ),
    ReasonInfo(
        MERCHANDISE_RETURN, RETAINED, ROLE_MERCHANDISE_RETURN,
        "Cancellation invoice with negative quantity that passed every rule above.",
    ),
    ReasonInfo(
        MERCHANDISE_SALE, RETAINED, ROLE_MERCHANDISE_SALE,
        "Row that passed every rule above.",
    ),
)

REASON_CODES: tuple[str, ...] = tuple(info.code for info in REASONS)
REASON_BY_CODE: dict[str, ReasonInfo] = {info.code: info for info in REASONS}
DISPOSITION_BY_CODE: dict[str, str] = {i.code: i.disposition for i in REASONS}
ROLE_BY_CODE: dict[str, str] = {i.code: i.role for i in REASONS}