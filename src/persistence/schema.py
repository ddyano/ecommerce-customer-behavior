"""Versioned validation and in-memory persistence views for annotated data.

This module defines the boundary between the cleaner's annotated frame and a
storage-friendly representation. It does not write files or alter source data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.persistence.hashing import canonical_json_bytes, sha256_hex
from src.preprocessing import rules

SCHEMA_VERSION = 1

# This order is the annotated output produced for the repository's configured
# eight input columns. It is intentionally pinned independently of config.
ANNOTATED_COLUMNS: tuple[str, ...] = (
    "row_id",
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "source_sheet",
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

RULE_CATEGORICALS: dict[str, tuple[str, ...]] = {
    "reason_code": rules.REASON_CODES,
    "disposition": (rules.RETAINED, rules.EXCLUDED),
    "analytic_role": rules.ROLES,
}
DYNAMIC_CATEGORICALS: tuple[str, ...] = ("Country", "source_sheet")
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "reason_code",
    "disposition",
    "analytic_role",
    *DYNAMIC_CATEGORICALS,
)

_RAW_COLUMNS: tuple[str, ...] = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "source_sheet",
)
_NON_NULL_COLUMNS = frozenset(
    {
        "row_id",
        "source_sheet",
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
    }
)
_BOOLEAN_COLUMNS = (
    "has_customer_id",
    "is_cancellation_invoice",
    "is_voucher_code",
    "is_exact_duplicate",
    "is_merchandise_sale",
    "is_merchandise_return",
    "is_voucher",
    "is_excluded",
    "is_basket_eligible",
)
_FLOAT_COLUMNS = ("quantity_num", "price_num", "line_value")
_STRING_COLUMNS = ("reason_code", "disposition", "analytic_role")
_DYNAMIC_TEXT_COLUMNS = ("Country", "source_sheet")
_RAW_TEXT_COLUMNS = ("Invoice", "StockCode", "Description", "Country", "source_sheet")
_RAW_NUMBER_COLUMNS = ("Quantity", "Price", "Customer ID")
_RAW_DATETIME_COLUMNS = ("InvoiceDate",)
_RAW_DTYPE_FAMILIES: dict[str, frozenset[str]] = {
    **{name: frozenset({"object", "string", "categorical"}) for name in _RAW_TEXT_COLUMNS},
    **{name: frozenset({"object", "string", "int", "float"}) for name in _RAW_NUMBER_COLUMNS},
    **{name: frozenset({"object", "string", "datetime"}) for name in _RAW_DATETIME_COLUMNS},
}
_RAW_DTYPE_FAMILIES["Country"] = frozenset({"object", "string"})
_RAW_DTYPE_FAMILIES["source_sheet"] = frozenset({"object", "string"})


class SchemaError(ValueError):
    """An annotated or persistence view violates the versioned schema."""


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    nullable: bool


@dataclass(frozen=True)
class PersistenceView:
    """In-memory persistence form plus untrusted, validated restoration metadata.

    The frame contains categorical columns. Metadata describes the original
    dtypes and the dynamic category vocabularies needed to restore it.
    """

    frame: pd.DataFrame
    schema_version: int
    original_dtypes: tuple[tuple[str, dict[str, Any]], ...]
    dynamic_categories: tuple[tuple[str, tuple[str, ...]], ...]
    dynamic_nulls: tuple[tuple[str, tuple[tuple[int, str], ...]], ...]


def annotated_schema() -> tuple[ColumnSpec, ...]:
    """Return the ordered version-1 column contract.

    Raw columns use supported dtype families recorded exactly at conversion
    time; derived columns use the fixed dtypes emitted by the cleaner, except
    invoice_datetime which accepts any naive datetime64 unit.
    """
    specs: list[ColumnSpec] = []
    for name in ANNOTATED_COLUMNS:
        if name in _RAW_COLUMNS:
            dtype = "supported raw dtype (recorded exactly)"
        elif name == "row_id":
            dtype = "int64"
        elif name in _BOOLEAN_COLUMNS:
            dtype = "bool"
        elif name in _FLOAT_COLUMNS:
            dtype = "float64"
        elif name == "invoice_datetime":
            dtype = "naive datetime64 (unit preserved)"
        elif name in _STRING_COLUMNS:
            dtype = "pandas string"
        else:  # defensive: schema table and contract must stay in lockstep
            raise SchemaError(f"No dtype rule is defined for column {name!r}.")
        specs.append(ColumnSpec(name, dtype, name not in _NON_NULL_COLUMNS))
    return tuple(specs)


def _dtype_family(dtype: Any) -> str:
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if isinstance(dtype, pd.StringDtype):
        return "string"
    if isinstance(dtype, np.dtype):
        if dtype.kind == "O":
            return "object"
        if dtype.kind == "b":
            return "bool"
        if dtype.kind in "iu":
            return "int"
        if dtype.kind == "f":
            return "float"
        if dtype.kind == "M":
            return "datetime"
    return "unsupported"


def _category_value(value: Any) -> bool:
    return type(value) in (str, int, float, bool) and not (
        type(value) is float and not math.isfinite(value)
    )


def _dtype_metadata(dtype: Any, column: str) -> dict[str, Any]:
    """Build a JSON-safe exact dtype descriptor; reject unsupported dtypes."""
    family = _dtype_family(dtype)
    if family == "object":
        return {"kind": "object"}
    if family == "string":
        na_kind = "pd.NA" if dtype.na_value is pd.NA else "nan" if (
            isinstance(dtype.na_value, float) and math.isnan(dtype.na_value)
        ) else None
        if na_kind is None:
            raise SchemaError(f"column {column!r} has unsupported string missing-value metadata.")
        return {
            "kind": "string",
            "name": dtype.name,
            "storage": dtype.storage,
            "na_value": na_kind,
        }
    if family == "categorical":
        categories = dtype.categories.tolist()
        if not all(_category_value(value) for value in categories):
            raise SchemaError(f"column {column!r} has unsupported categorical values.")
        return {
            "kind": "categorical",
            "categories": categories,
            "ordered": bool(dtype.ordered),
        }
    if family in {"bool", "int", "float", "datetime"}:
        return {"kind": family, "name": dtype.name}
    raise SchemaError(f"column {column!r} has unsupported dtype {dtype}.")


def _validate_object_values(series: pd.Series, column: str) -> None:
    """Restrict object payloads to the exact scalar types supported by hashing."""
    supported = (str, int, float, bool, bytes)
    for value in series.array:
        if value is None or value is pd.NA:
            continue
        if type(value) not in supported and not (
            type(value) is float and math.isnan(value)
        ):
            raise SchemaError(f"column {column!r} contains an unsupported object value type.")


def _metadata_dtype(spec: Any, column: str) -> Any:
    """Parse and strictly validate an untrusted dtype descriptor."""
    if not isinstance(spec, dict) or not isinstance(spec.get("kind"), str):
        raise SchemaError(f"dtype metadata for {column!r} is missing or malformed.")
    kind = spec["kind"]
    try:
        if kind == "object" and set(spec) == {"kind"}:
            return np.dtype("O")
        if kind == "string" and set(spec) == {"kind", "name", "storage", "na_value"}:
            if spec["name"] not in {"string", "str"}:
                raise ValueError
            if spec["storage"] not in {"python", "pyarrow"}:
                raise ValueError
            na_value = {"pd.NA": pd.NA, "nan": np.nan}.get(spec["na_value"])
            if na_value is None:
                raise ValueError
            dtype = pd.StringDtype(storage=spec["storage"], na_value=na_value)
            if dtype.name != spec["name"]:
                raise ValueError
            return dtype
        if kind in {"bool", "int", "float", "datetime"} and set(spec) == {"kind", "name"}:
            if not isinstance(spec["name"], str):
                raise ValueError
            dtype = np.dtype(spec["name"])
            actual = _dtype_family(dtype)
            if actual != kind or (kind == "datetime" and dtype.kind != "M"):
                raise ValueError
            return dtype
        if kind == "categorical" and set(spec) == {"kind", "categories", "ordered"}:
            categories = spec["categories"]
            ordered = spec["ordered"]
            if not isinstance(categories, list) or type(ordered) is not bool:
                raise ValueError
            if not all(_category_value(value) for value in categories):
                raise ValueError
            if len({(type(value), value) for value in categories}) != len(categories):
                raise ValueError
            return pd.CategoricalDtype(categories=categories, ordered=ordered)
    except (TypeError, ValueError, OverflowError):
        pass
    raise SchemaError(f"dtype metadata for {column!r} is malformed or unsupported.")


def _check_frame_columns(frame: Any) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError("Expected a pandas DataFrame.")
    names = list(frame.columns)
    if not all(isinstance(name, str) for name in names):
        raise SchemaError("DataFrame column names must be strings.")
    if len(names) != len(set(names)):
        raise SchemaError("DataFrame has duplicate column names.")
    if tuple(names) != ANNOTATED_COLUMNS:
        raise SchemaError("DataFrame columns do not match the ordered version-1 schema.")
    return frame


def _check_nullability(frame: pd.DataFrame) -> None:
    for name in _NON_NULL_COLUMNS:
        try:
            has_null = bool(frame[name].isna().any())
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"column {name!r} cannot be checked for null values.") from exc
        if has_null:
            raise SchemaError(f"column {name!r} contains forbidden null values.")


def _check_categorical_labels(frame: pd.DataFrame) -> None:
    for column, allowed in RULE_CATEGORICALS.items():
        values = frame[column]
        if values.isna().any():
            raise SchemaError(f"column {column!r} contains forbidden null values.")
        invalid = ~values.isin(allowed)
        if bool(invalid.any()):
            raise SchemaError(f"column {column!r} contains values outside its rule vocabulary.")
    for column in _DYNAMIC_TEXT_COLUMNS:
        values = frame[column]
        if column == "source_sheet" and values.isna().any():
            raise SchemaError("column 'source_sheet' contains forbidden null values.")
        for value in values.dropna().array:
            if type(value) is not str:
                raise SchemaError(f"column {column!r} must contain only strings or nulls.")


def _check_annotated_dtypes(frame: pd.DataFrame) -> None:
    for name in _RAW_COLUMNS:
        dtype = frame[name].dtype
        family = _dtype_family(dtype)
        if family not in _RAW_DTYPE_FAMILIES[name]:
            raise SchemaError(f"column {name!r} has unsupported raw dtype {dtype}.")
        _dtype_metadata(dtype, name)
        if family == "object":
            _validate_object_values(frame[name], name)
    if frame["row_id"].dtype != np.dtype("int64"):
        raise SchemaError("column 'row_id' must have dtype int64.")
    for name in _BOOLEAN_COLUMNS:
        if frame[name].dtype != np.dtype("bool"):
            raise SchemaError(f"column {name!r} must have dtype bool.")
    for name in _FLOAT_COLUMNS:
        if frame[name].dtype != np.dtype("float64"):
            raise SchemaError(f"column {name!r} must have dtype float64.")
    if not isinstance(frame["invoice_datetime"].dtype, np.dtype) or frame["invoice_datetime"].dtype.kind != "M":
        raise SchemaError("column 'invoice_datetime' must have naive datetime64 dtype.")
    if getattr(frame["invoice_datetime"].dtype, "tz", None) is not None:
        raise SchemaError("column 'invoice_datetime' must be timezone-naive.")
    for name in _STRING_COLUMNS:
        if not isinstance(frame[name].dtype, pd.StringDtype):
            raise SchemaError(f"column {name!r} must have pandas string dtype.")


def validate_annotated_view(frame: pd.DataFrame) -> None:
    """Validate an annotated DataFrame against schema version 1."""
    _check_frame_columns(frame)
    _check_annotated_dtypes(frame)
    _check_nullability(frame)
    _check_categorical_labels(frame)


def _dynamic_category_values(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    categories: dict[str, tuple[str, ...]] = {}
    for name in DYNAMIC_CATEGORICALS:
        values = [value for value in frame[name].tolist() if not pd.isna(value)]
        categories[name] = tuple(sorted(set(values)))
    return categories


def _dynamic_null_metadata(
    frame: pd.DataFrame,
) -> tuple[tuple[str, tuple[tuple[int, str], ...]], ...]:
    """Record object-column missing sentinels lost by categorical encoding."""
    result: list[tuple[str, tuple[tuple[int, str], ...]]] = []
    for name in DYNAMIC_CATEGORICALS:
        entries: list[tuple[int, str]] = []
        if _dtype_family(frame[name].dtype) == "object":
            for position, value in enumerate(frame[name].tolist()):
                if value is None:
                    entries.append((position, "none"))
                elif value is pd.NA:
                    entries.append((position, "pd.NA"))
                elif type(value) is float and math.isnan(value):
                    entries.append((position, "nan"))
        result.append((name, tuple(entries)))
    return tuple(result)


def _expected_dtype_metadata(frame: pd.DataFrame) -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple((name, _dtype_metadata(frame[name].dtype, name)) for name in ANNOTATED_COLUMNS)


def to_persistence_view(frame: pd.DataFrame) -> PersistenceView:
    """Convert a validated annotated frame to an in-memory categorical view."""
    validate_annotated_view(frame)
    original_dtypes = _expected_dtype_metadata(frame)
    dynamic = _dynamic_category_values(frame)
    result = frame.copy(deep=True)
    categories_by_column: dict[str, tuple[str, ...]] = {
        **RULE_CATEGORICALS,
        **dynamic,
    }
    for name in CATEGORICAL_COLUMNS:
        result[name] = pd.Series(
            pd.Categorical(
                frame[name].array,
                categories=list(categories_by_column[name]),
                ordered=False,
            ),
            index=frame.index,
            name=name,
        )
    return PersistenceView(
        frame=result,
        schema_version=SCHEMA_VERSION,
        original_dtypes=original_dtypes,
        dynamic_categories=tuple((name, dynamic[name]) for name in DYNAMIC_CATEGORICALS),
        dynamic_nulls=_dynamic_null_metadata(frame),
    )


def _validate_metadata(view: Any) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    if not isinstance(view, PersistenceView):
        raise SchemaError("Expected a PersistenceView with restoration metadata.")
    if type(view.schema_version) is not int or view.schema_version != SCHEMA_VERSION:
        raise SchemaError("Persistence view has a missing or unsupported schema version.")
    if not isinstance(view.original_dtypes, tuple) or len(view.original_dtypes) != len(ANNOTATED_COLUMNS):
        raise SchemaError("Persistence view dtype metadata is missing or malformed.")
    dtypes: dict[str, Any] = {}
    for expected_name, item in zip(ANNOTATED_COLUMNS, view.original_dtypes):
        if not isinstance(item, tuple) or len(item) != 2 or item[0] != expected_name:
            raise SchemaError("Persistence view dtype metadata has inconsistent columns.")
        dtypes[expected_name] = _metadata_dtype(item[1], expected_name)

    if not isinstance(view.dynamic_categories, tuple) or len(view.dynamic_categories) != len(DYNAMIC_CATEGORICALS):
        raise SchemaError("Persistence view dynamic category metadata is missing or malformed.")
    dynamic: dict[str, tuple[str, ...]] = {}
    for expected_name, item in zip(DYNAMIC_CATEGORICALS, view.dynamic_categories):
        if not isinstance(item, tuple) or len(item) != 2 or item[0] != expected_name:
            raise SchemaError("Persistence view dynamic category metadata has inconsistent columns.")
        categories = item[1]
        if (
            not isinstance(categories, tuple)
            or not all(type(value) is str for value in categories)
            or tuple(sorted(set(categories))) != categories
        ):
            raise SchemaError(f"dynamic category metadata for {expected_name!r} is malformed.")
        dynamic[expected_name] = categories
    if not isinstance(view.dynamic_nulls, tuple) or len(view.dynamic_nulls) != len(DYNAMIC_CATEGORICALS):
        raise SchemaError("Persistence view dynamic null metadata is missing or malformed.")
    for expected_name, item in zip(DYNAMIC_CATEGORICALS, view.dynamic_nulls):
        if not isinstance(item, tuple) or len(item) != 2 or item[0] != expected_name:
            raise SchemaError("Persistence view dynamic null metadata has inconsistent columns.")
        entries = item[1]
        if not isinstance(entries, tuple):
            raise SchemaError(f"dynamic null metadata for {expected_name!r} is malformed.")
        previous = -1
        for entry in entries:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or type(entry[0]) is not int
                or entry[0] <= previous
                or type(entry[1]) is not str
                or entry[1] not in {"none", "pd.NA", "nan"}
            ):
                raise SchemaError(f"dynamic null metadata for {expected_name!r} is malformed.")
            if entry[0] >= len(view.frame):
                raise SchemaError(f"dynamic null metadata for {expected_name!r} is out of range.")
            previous = entry[0]
        if entries and _dtype_family(dtypes[expected_name]) != "object":
            raise SchemaError(f"dynamic null metadata for {expected_name!r} conflicts with its dtype.")
        if expected_name == "source_sheet" and entries:
            raise SchemaError("source_sheet cannot have dynamic null metadata.")
        if _dtype_family(dtypes[expected_name]) == "object":
            category_series = view.frame[expected_name]
            null_positions = tuple(np.flatnonzero(category_series.isna().to_numpy()).tolist())
            if tuple(position for position, _ in entries) != null_positions:
                raise SchemaError(f"dynamic null metadata for {expected_name!r} is inconsistent with values.")
    return dtypes, dynamic


def validate_persistence_view(view: PersistenceView) -> None:
    """Validate the persistence representation and its untrusted metadata."""
    if not isinstance(view, PersistenceView):
        raise SchemaError("Expected a PersistenceView with restoration metadata.")
    frame = _check_frame_columns(view.frame)
    dtypes, dynamic = _validate_metadata(view)
    if frame["row_id"].dtype != np.dtype("int64"):
        raise SchemaError("column 'row_id' must have dtype int64.")
    for name in _BOOLEAN_COLUMNS:
        if frame[name].dtype != np.dtype("bool"):
            raise SchemaError(f"column {name!r} must have dtype bool.")
    for name in _FLOAT_COLUMNS:
        if frame[name].dtype != np.dtype("float64"):
            raise SchemaError(f"column {name!r} must have dtype float64.")
    if not isinstance(frame["invoice_datetime"].dtype, np.dtype) or frame["invoice_datetime"].dtype.kind != "M" or getattr(frame["invoice_datetime"].dtype, "tz", None) is not None:
        raise SchemaError("column 'invoice_datetime' must have naive datetime64 dtype.")
    for name in _RAW_COLUMNS:
        if name in CATEGORICAL_COLUMNS:
            continue
        if _dtype_family(frame[name].dtype) not in _RAW_DTYPE_FAMILIES[name]:
            raise SchemaError(f"column {name!r} has unsupported raw dtype in persistence view.")
        if _dtype_metadata(frame[name].dtype, name) != _dtype_metadata(dtypes[name], name):
            raise SchemaError(f"column {name!r} dtype disagrees with restoration metadata.")

    categories_by_column: dict[str, tuple[str, ...]] = {
        **RULE_CATEGORICALS,
        **dynamic,
    }
    for name in CATEGORICAL_COLUMNS:
        series = frame[name]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            raise SchemaError(f"column {name!r} must be categorical in the persistence view.")
        if series.dtype.ordered:
            raise SchemaError(f"column {name!r} categorical must be unordered.")
        if tuple(series.cat.categories.tolist()) != categories_by_column[name]:
            raise SchemaError(f"column {name!r} categories disagree with the schema metadata.")
        if name in RULE_CATEGORICALS:
            if dtypes[name] != pd.StringDtype():
                raise SchemaError(f"restoration dtype metadata for {name!r} is invalid.")
        elif _dtype_family(dtypes[name]) not in {"object", "string"}:
            raise SchemaError(f"restoration dtype metadata for {name!r} is unsupported.")
        if name in _NON_NULL_COLUMNS and series.isna().any():
            raise SchemaError(f"column {name!r} contains forbidden null values.")
        codes = series.cat.codes.to_numpy()
        if bool((codes >= len(categories_by_column[name])).any()):
            raise SchemaError(f"column {name!r} contains invalid categorical codes.")
        if name in DYNAMIC_CATEGORICALS:
            observed = tuple(sorted(set(series.dropna().tolist())))
            if observed != dynamic[name]:
                raise SchemaError(f"column {name!r} values disagree with dynamic categories.")

    # For non-categorical fields the dtype metadata must match the live frame.
    for name in ANNOTATED_COLUMNS:
        if name not in CATEGORICAL_COLUMNS and _dtype_metadata(frame[name].dtype, name) != _dtype_metadata(dtypes[name], name):
            raise SchemaError(f"column {name!r} dtype disagrees with restoration metadata.")
    _check_nullability(frame)


def to_annotated_view(view: PersistenceView) -> pd.DataFrame:
    """Restore a persistence view, independently validating all categories."""
    # This validation is deliberately unconditional: callers cannot bypass it
    # by invoking restoration without first calling validate_persistence_view.
    validate_persistence_view(view)
    dtypes, _ = _validate_metadata(view)
    restored = view.frame.copy(deep=True)
    for name in CATEGORICAL_COLUMNS:
        restored[name] = restored[name].astype(dtypes[name])
    nulls = dict(view.dynamic_nulls)
    for name in DYNAMIC_CATEGORICALS:
        if _dtype_family(dtypes[name]) != "object":
            continue
        series = restored[name].astype(object)
        for position, marker in nulls[name]:
            if position >= len(series) or not pd.isna(series.iloc[position]):
                raise SchemaError(f"dynamic null metadata for {name!r} is inconsistent with values.")
            series.iloc[position] = {"none": None, "pd.NA": pd.NA, "nan": np.nan}[marker]
        restored[name] = series
    validate_annotated_view(restored)
    return restored


def schema_digest() -> str:
    """Return a deterministic digest of the dataset-independent schema."""
    definition = {
        "schema_version": SCHEMA_VERSION,
        "columns": [
            {"name": spec.name, "dtype": spec.dtype, "nullable": spec.nullable}
            for spec in annotated_schema()
        ],
        "raw_dtype_families": {
            name: sorted(_RAW_DTYPE_FAMILIES[name]) for name in sorted(_RAW_DTYPE_FAMILIES)
        },
        "rule_categories": {name: list(values) for name, values in RULE_CATEGORICALS.items()},
        "dynamic_categories": list(DYNAMIC_CATEGORICALS),
        "categorical_storage": "unordered",
    }
    return sha256_hex(canonical_json_bytes(definition))
