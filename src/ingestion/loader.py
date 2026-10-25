"""Load the raw retail dataset from a local .xlsx or .csv file.

This module only LOADS and CHECKS the file. It does not clean, filter or change
any values: cancellations, missing customer IDs and odd quantities are kept
exactly as found. Cleaning happens in a later step.

Local-only: reads the one file it is given and makes no network calls.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES: tuple[str, ...] = (".xlsx", ".csv")
MAX_FILE_BYTES: int = 500 * 1024 * 1024
# Read as text so codes like "C489449" or "15056BL" are never mangled.
TEXT_COLUMNS: dict[str, type] = {"Invoice": str, "StockCode": str}
SOURCE_COLUMN: str = "source_sheet"
CSV_SOURCE_LABEL: str = "csv"


class DataLoadError(Exception):
    """Raised when the data file cannot be found, read or used."""


class SchemaError(DataLoadError):
    """Raised when the file does not contain the expected columns."""


def normalize_columns(columns: Iterable[object]) -> list[str]:
    """Strip whitespace from column names and reject duplicates."""
    cleaned = [str(column).strip() for column in columns]
    duplicates = sorted({name for name in cleaned if cleaned.count(name) > 1})
    if duplicates:
        raise SchemaError(f"Duplicate column names after trimming: {duplicates}")
    return cleaned


def validate_schema(
    columns: Sequence[str], required_columns: Sequence[str], label: str = "data"
) -> None:
    """Raise SchemaError if any required column is missing."""
    missing = [name for name in required_columns if name not in columns]
    if missing:
        raise SchemaError(
            f"{label}: missing required column(s) {missing}. "
            f"Columns found: {list(columns)}"
        )


def _check_file(path: Path | str) -> Path:
    file_path = Path(path)
    if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise DataLoadError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Allowed: {list(ALLOWED_SUFFIXES)}"
        )
    if not file_path.is_file():
        raise DataLoadError(f"File not found: {file_path.name}")
    size = file_path.stat().st_size
    if size == 0:
        raise DataLoadError(f"File is empty: {file_path.name}")
    if size > MAX_FILE_BYTES:
        raise DataLoadError(
            f"File is too large ({size} bytes; limit {MAX_FILE_BYTES}): "
            f"{file_path.name}"
        )
    return file_path


def _read_sheets(
    file_path: Path, sheet_names: Sequence[str] | None
) -> dict[str, pd.DataFrame]:
    try:
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path, dtype=TEXT_COLUMNS, encoding="utf-8")
            return {CSV_SOURCE_LABEL: frame}
        selected = list(sheet_names) if sheet_names else None
        return pd.read_excel(
            file_path, sheet_name=selected, engine="openpyxl", dtype=TEXT_COLUMNS
        )
    except Exception as exc:  # noqa: BLE001 - convert any read failure to one clear error
        raise DataLoadError(
            f"Could not read {file_path.name}: {type(exc).__name__}: {exc}"
        ) from exc


def load_raw_data(
    path: Path | str,
    required_columns: Sequence[str],
    sheet_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a retail file, check its columns, and return all rows unchanged.

    Excel workbooks: every sheet (or only ``sheet_names``) is read and stacked.
    A ``source_sheet`` column records where each row came from ("csv" for CSV).
    """
    file_path = _check_file(path)
    sheets = _read_sheets(file_path, sheet_names)

    frames: list[pd.DataFrame] = []
    for sheet_label, frame in sheets.items():
        label = f"{file_path.name} [{sheet_label}]"
        frame.columns = normalize_columns(frame.columns)
        validate_schema(list(frame.columns), required_columns, label)
        if SOURCE_COLUMN in frame.columns:
            raise DataLoadError(
                f"{label}: the column name '{SOURCE_COLUMN}' is reserved."
            )
        if frame.empty:
            logger.warning("%s has no data rows and was skipped.", label)
            continue
        frame[SOURCE_COLUMN] = str(sheet_label)
        frames.append(frame)

    if not frames:
        raise DataLoadError(f"{file_path.name} contains no data rows.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %s.", len(combined), file_path.name)
    return combined