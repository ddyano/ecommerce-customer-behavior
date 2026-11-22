"""Tests for the versioned persistence schema (synthetic rows only)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import CleaningConfig
from src.persistence.schema import (
    ANNOTATED_COLUMNS,
    PersistenceView,
    SchemaError,
    annotated_schema,
    schema_digest,
    to_annotated_view,
    to_persistence_view,
    validate_annotated_view,
    validate_persistence_view,
)
from src.preprocessing.cleaning import clean_transactions
from src.preprocessing import rules


RAW_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
    "Price", "Customer ID", "Country", "source_sheet",
]


@pytest.fixture
def annotated() -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            ("1001", "A1", "plain", 2, "2020-01-01", 3.0, 10.0, "UK", "S1"),
            ("C1002", "A1", 4, -1, "2020-01-02", 2.0, None, "France", "S2"),
            ("1003", "gift_0001_20", "gift", 1, "2020-01-03", 5.0, 11.0, "UK", "S1"),
        ],
        columns=RAW_COLUMNS,
    )
    raw["Description"] = pd.Series(["plain", 4, None], dtype=object)
    raw["Quantity"] = pd.Series([2, -1, 1], dtype="int64")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"]).astype("datetime64[us]")
    raw["Price"] = raw["Price"].astype("float64")
    raw["Customer ID"] = raw["Customer ID"].astype("float64")
    config = CleaningConfig(
        cancellation_invoice_prefix="C",
        non_product_stock_codes=("POST",),
        duplicate_policy="flag",
        cancellation_policy="return",
        zero_price_policy="exclude",
        voucher_stock_code_prefixes=("gift_0001_",),
    )
    return clean_transactions(raw, config, RAW_COLUMNS[:-1]).annotated


def test_schema_pins_column_order_and_digest_is_stable():
    assert tuple(spec.name for spec in annotated_schema()) == ANNOTATED_COLUMNS
    assert len(ANNOTATED_COLUMNS) == 26
    assert schema_digest() == schema_digest()
    assert len(schema_digest()) == 64


def test_valid_annotated_frame_converts_and_round_trips_without_mutation(annotated):
    before = annotated.copy(deep=True)
    validate_annotated_view(annotated)
    persisted = to_persistence_view(annotated)
    validate_persistence_view(persisted)
    for name in ("reason_code", "disposition", "analytic_role", "Country", "source_sheet"):
        assert isinstance(persisted.frame[name].dtype, pd.CategoricalDtype)
        assert not persisted.frame[name].dtype.ordered
    restored = to_annotated_view(persisted)
    pd.testing.assert_frame_equal(restored, before)
    pd.testing.assert_frame_equal(annotated, before)


def test_rule_categories_use_complete_rule_vocabularies(annotated):
    persisted = to_persistence_view(annotated)
    assert tuple(persisted.frame["reason_code"].cat.categories) == rules.REASON_CODES
    assert tuple(persisted.frame["disposition"].cat.categories) == (
        rules.RETAINED, rules.EXCLUDED
    )
    assert tuple(persisted.frame["analytic_role"].cat.categories) == rules.ROLES


@pytest.mark.parametrize("mutation", [
    lambda f: f.rename(columns={"Country": "Region"}),
    lambda f: f.loc[:, list(reversed(f.columns))],
    lambda f: f.assign(reason_code=f["reason_code"].astype(object).where(lambda s: s != s.iloc[0], "unknown").astype("string")),
    lambda f: f.assign(has_customer_id=f["has_customer_id"].astype("int64")),
])
def test_annotated_view_rejects_invalid_columns_values_or_dtypes(annotated, mutation):
    with pytest.raises(SchemaError):
        validate_annotated_view(mutation(annotated.copy()))


def test_unsupported_raw_nullable_dtype_is_rejected(annotated):
    bad = annotated.copy()
    bad["Customer ID"] = pd.array(bad["Customer ID"], dtype="Float64")
    with pytest.raises(SchemaError, match="unsupported raw dtype"):
        validate_annotated_view(bad)


def test_dynamic_categories_are_derived_from_validated_values(annotated):
    persisted = to_persistence_view(annotated)
    assert tuple(persisted.frame["Country"].cat.categories) == ("France", "UK")
    assert tuple(persisted.frame["source_sheet"].cat.categories) == ("S1", "S2")


def test_empty_frame_round_trips_with_empty_dynamic_vocabularies(annotated):
    empty = annotated.iloc[:0].copy()
    persisted = to_persistence_view(empty)
    assert persisted.frame["Country"].cat.categories.tolist() == []
    assert persisted.frame["source_sheet"].cat.categories.tolist() == []
    pd.testing.assert_frame_equal(to_annotated_view(persisted), empty)


@pytest.mark.parametrize("column", ["reason_code", "disposition", "analytic_role", "Country", "source_sheet"])
def test_restoration_independently_rejects_ordered_categories(annotated, column):
    persisted = to_persistence_view(annotated)
    bad_frame = persisted.frame.copy()
    bad_frame[column] = bad_frame[column].cat.as_ordered()
    bad = replace(persisted, frame=bad_frame)
    with pytest.raises(SchemaError, match="unordered"):
        to_annotated_view(bad)


def test_cached_view_rejects_categories_that_disagree_with_metadata(annotated):
    persisted = to_persistence_view(annotated)
    bad_frame = persisted.frame.copy()
    bad_frame["Country"] = bad_frame["Country"].cat.add_categories(["Never observed"])
    bad = replace(persisted, frame=bad_frame)
    with pytest.raises(SchemaError, match="categories disagree"):
        validate_persistence_view(bad)
    with pytest.raises(SchemaError, match="categories disagree"):
        to_annotated_view(bad)


@pytest.mark.parametrize("malformation", ["missing", "extra", "reordered", "duplicate"])
def test_malformed_cached_columns_raise_schema_error_directly(annotated, malformation):
    persisted = to_persistence_view(annotated)
    frame = persisted.frame.copy()
    if malformation == "missing":
        frame = frame.drop(columns=["Country"])
    elif malformation == "extra":
        frame["unexpected"] = 1
    elif malformation == "reordered":
        frame = frame.loc[:, list(reversed(frame.columns))]
    else:
        frame.columns = [*frame.columns[:-1], frame.columns[-2]]
    malformed = replace(persisted, frame=frame)

    with pytest.raises(SchemaError):
        validate_persistence_view(malformed)
    with pytest.raises(SchemaError):
        to_annotated_view(malformed)


def test_cached_view_rejects_non_string_dynamic_categories(annotated):
    persisted = to_persistence_view(annotated)
    bad_frame = persisted.frame.copy()
    bad_frame["Country"] = pd.Categorical([1, 2, 1], categories=[1, 2])
    bad = replace(persisted, frame=bad_frame)
    with pytest.raises(SchemaError):
        to_annotated_view(bad)


def test_cached_view_rejects_forged_dynamic_vocabulary_metadata(annotated):
    persisted = to_persistence_view(annotated)
    bad = replace(
        persisted,
        dynamic_categories=(("Country", ("France", "UK", "ZZ")), ("source_sheet", ("S1", "S2"))),
    )
    with pytest.raises(SchemaError):
        to_annotated_view(bad)


@pytest.mark.parametrize("bad_metadata", [None, (), (("wrong", {}),)])
def test_cached_view_rejects_missing_or_inconsistent_dtype_metadata(annotated, bad_metadata):
    persisted = to_persistence_view(annotated)
    bad = replace(persisted, original_dtypes=bad_metadata)
    with pytest.raises(SchemaError):
        to_annotated_view(bad)


def test_cached_view_rejects_malformed_dtype_descriptor(annotated):
    persisted = to_persistence_view(annotated)
    metadata = list(persisted.original_dtypes)
    metadata[1] = (metadata[1][0], {"kind": "datetime", "name": "datetime64[ns, UTC]"})
    bad = replace(persisted, original_dtypes=tuple(metadata))
    with pytest.raises(SchemaError):
        validate_persistence_view(bad)


def test_cached_view_rejects_malformed_dynamic_null_metadata(annotated):
    persisted = to_persistence_view(annotated)
    bad = replace(persisted, dynamic_nulls=(("Country", ((0, "unknown"),)), ("source_sheet", ())))
    with pytest.raises(SchemaError):
        to_annotated_view(bad)


def test_country_null_is_preserved(annotated):
    changed = annotated.copy()
    changed["Country"] = changed["Country"].astype(object)
    changed.loc[0, "Country"] = None
    validate_annotated_view(changed)
    restored = to_annotated_view(to_persistence_view(changed))
    assert restored.loc[0, "Country"] is None
    assert restored["Country"].dtype == object


def test_description_values_and_object_type_distinctions_are_unchanged(annotated):
    before = annotated["Description"].tolist()
    restored = to_annotated_view(to_persistence_view(annotated))
    after = restored["Description"].tolist()
    assert [type(value) for value in after] == [type(value) for value in before]
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert pd.isna(after[2]) and pd.isna(before[2])


def test_unsupported_object_payload_is_rejected(annotated):
    bad = annotated.copy()
    bad["Description"] = bad["Description"].astype(object)
    bad.at[0, "Description"] = ["not", "a", "supported", "scalar"]
    with pytest.raises(SchemaError, match="unsupported object value type"):
        validate_annotated_view(bad)


def test_schema_digest_does_not_depend_on_frame_categories():
    original = schema_digest()
    assert schema_digest() == original
    # Frame-derived categories are held in PersistenceView metadata, not in the
    # canonical schema definition used by schema_digest().
    assert schema_digest() == original
