"""Checks that the REAL configs/config.yaml decisions take effect in cleaning.

The rows are SYNTHETIC (invented). Only the configuration is real.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.preprocessing.cleaning import clean_transactions

COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "source_sheet",
]

# (stock code, reason the approved decisions should give it)
CASES = [
    ("TEST001", "listed_non_product_code"),
    ("TEST002", "listed_non_product_code"),
    ("POST", "listed_non_product_code"),
    ("DCGSSGIRL", "listed_unidentifiable_code"),
    ("gift_0001_20", "voucher_sale"),
    ("DCGS0058", "merchandise_sale"),
    ("PADS", "merchandise_sale"),
    ("85123A", "merchandise_sale"),
]


def test_real_config_decisions_take_effect_in_cleaning():
    config = load_config()
    rows = [
        (
            str(1000 + position),
            code,
            "SYNTHETIC ITEM",
            1,
            "2020-01-01 10:00",
            2.0,
            1.0,
            "UK",
            "S1",
        )
        for position, (code, _) in enumerate(CASES)
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    cleaned = clean_transactions(
        frame, config.cleaning, config.schema.required_columns
    )
    assert cleaned.annotated["reason_code"].tolist() == [
        reason for _, reason in CASES
    ]