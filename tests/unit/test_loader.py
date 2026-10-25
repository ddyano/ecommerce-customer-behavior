"""Tests for src.ingestion.loader.

All data here is SYNTHETIC (invented rows written to temporary files). None of
it comes from the real Online Retail II dataset.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion import loader as loader_module
from src.ingestion.loader import (
    DataLoadError,
    SchemaError,
    load_raw_data,
    normalize_columns,
    validate_schema,
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


def _rows() -> pd.DataFrame:
    """Three invented rows: a normal sale, a cancellation, and an odd row."""
    return pd.DataFrame(
        {
            "Invoice": ["489434", "C489449", "489435"],
            "StockCode": ["00123", "15056BL", "85048"],
            "Description": ["SYNTHETIC ITEM A", "SYNTHETIC ITEM B", None],
            "Quantity": [12, -1, 0],
            "InvoiceDate": [
                "2009-12-01 07:45:00",
                "2009-12-01 09:00:00",
                "2009-12-02 10:00:00",
            ],
            "Price": [6.95, 0.0, 2.5],
            "Customer ID": [13085.0, None, 13085.0],
            "Country": ["United Kingdom", "United Kingdom", "France"],
        }
    )


def _write_csv(tmp_path, frame: pd.DataFrame, name: str = "data.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _write_excel(tmp_path, sheets: dict, name: str = "data.xlsx"):
    path = tmp_path / name
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


# ----------------------------- helper functions -----------------------------
def test_normalize_columns_strips_and_stringifies():
    assert normalize_columns([" A ", "B", 3]) == ["A", "B", "3"]


def test_normalize_columns_rejects_duplicates():
    with pytest.raises(SchemaError, match="Duplicate"):
        normalize_columns(["A", " A"])


def test_validate_schema_passes_when_all_present():
    assert validate_schema(["a", "b", "c"], ["a", "b"]) is None


def test_validate_schema_names_missing_columns():
    with pytest.raises(SchemaError, match="b"):
        validate_schema(["a"], ["a", "b"])


# --------------------------------- CSV loading ---------------------------------
def test_csv_loads_all_rows_and_adds_source_column(tmp_path):
    loaded = load_raw_data(_write_csv(tmp_path, _rows()), REQUIRED)
    assert len(loaded) == 3
    assert list(loaded.columns) == REQUIRED + ["source_sheet"]
    assert set(loaded["source_sheet"]) == {"csv"}


def test_values_are_not_cleaned_or_changed(tmp_path):
    loaded = load_raw_data(_write_csv(tmp_path, _rows()), REQUIRED)
    assert loaded["Quantity"].tolist() == [12, -1, 0]
    assert loaded["Price"].tolist() == [6.95, 0.0, 2.5]
    assert int(loaded["Customer ID"].isna().sum()) == 1
    assert loaded["Invoice"].tolist() == ["489434", "C489449", "489435"]


def test_text_columns_keep_leading_zeros_and_letters(tmp_path):
    loaded = load_raw_data(_write_csv(tmp_path, _rows()), REQUIRED)
    assert loaded["StockCode"].tolist() == ["00123", "15056BL", "85048"]


def test_column_whitespace_is_trimmed(tmp_path):
    frame = _rows().rename(columns={"Invoice": " Invoice ", "Price": "Price "})
    loaded = load_raw_data(_write_csv(tmp_path, frame), REQUIRED)
    assert list(loaded.columns) == REQUIRED + ["source_sheet"]


# --------------------------------- Excel loading --------------------------------
def test_excel_sheets_are_stacked_with_source_label(tmp_path):
    path = _write_excel(tmp_path, {"A": _rows(), "B": _rows().iloc[:2]})
    loaded = load_raw_data(path, REQUIRED)
    assert len(loaded) == 5
    assert loaded["source_sheet"].tolist() == ["A", "A", "A", "B", "B"]


def test_excel_sheet_selection_reads_only_chosen_sheet(tmp_path):
    path = _write_excel(tmp_path, {"A": _rows(), "B": _rows().iloc[:2]})
    loaded = load_raw_data(path, REQUIRED, sheet_names=["B"])
    assert len(loaded) == 2
    assert set(loaded["source_sheet"]) == {"B"}


def test_excel_text_columns_keep_leading_zeros(tmp_path):
    path = _write_excel(tmp_path, {"A": _rows()})
    loaded = load_raw_data(path, REQUIRED)
    assert loaded["StockCode"].tolist() == ["00123", "15056BL", "85048"]


def test_excel_missing_column_in_one_sheet_names_that_sheet(tmp_path):
    path = _write_excel(
        tmp_path, {"A": _rows(), "B": _rows().drop(columns=["Price"])}
    )
    with pytest.raises(SchemaError, match=r"\[B\]"):
        load_raw_data(path, REQUIRED)


def test_excel_unknown_sheet_name_raises(tmp_path):
    path = _write_excel(tmp_path, {"A": _rows()})
    with pytest.raises(DataLoadError, match="Could not read"):
        load_raw_data(path, REQUIRED, sheet_names=["Nope"])


# ------------------------------- error handling -------------------------------
def test_missing_required_column_raises_schema_error(tmp_path):
    path = _write_csv(tmp_path, _rows().drop(columns=["Customer ID"]))
    with pytest.raises(SchemaError, match="Customer ID"):
        load_raw_data(path, REQUIRED)


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_raw_data(path, REQUIRED)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_raw_data(tmp_path / "nope.csv", REQUIRED)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(DataLoadError, match="empty"):
        load_raw_data(path, REQUIRED)


def test_header_only_file_raises(tmp_path):
    path = _write_csv(tmp_path, _rows().iloc[0:0])
    with pytest.raises(DataLoadError, match="no data rows"):
        load_raw_data(path, REQUIRED)


def test_oversized_file_raises(tmp_path, monkeypatch):
    path = _write_csv(tmp_path, _rows())
    monkeypatch.setattr(loader_module, "MAX_FILE_BYTES", 10)
    with pytest.raises(DataLoadError, match="too large"):
        load_raw_data(path, REQUIRED)


def test_corrupt_excel_file_raises(tmp_path):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"this is not a real workbook")
    with pytest.raises(DataLoadError, match="Could not read"):
        load_raw_data(path, REQUIRED)


def test_reserved_column_name_raises(tmp_path):
    frame = _rows()
    frame["source_sheet"] = "x"
    with pytest.raises(DataLoadError, match="reserved"):
        load_raw_data(_write_csv(tmp_path, frame), REQUIRED)