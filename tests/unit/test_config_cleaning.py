"""Tests for the cleaning-rule settings in src.config.

Config data is SYNTHETIC (temporary files), except one test that reads the
real configs/config.yaml to confirm your approved cleaning decisions are in it.
"""
from __future__ import annotations

import copy

import pytest
import yaml

from src.config import ConfigError, load_config

BASE = {
    "data": {
        "raw_path": "data/raw/sample.xlsx",
        "sheet_names": None,
        "processed_dir": "data/processed",
    },
    "schema": {"required_columns": ["Invoice", "Price"]},
    "cleaning": {
        "cancellation_invoice_prefix": "C",
        "non_product_stock_codes": ["POST"],
        "duplicate_policy": "flag",
    },
}


def _load(tmp_path, cleaning_updates):
    config = copy.deepcopy(BASE)
    config["cleaning"].update(cleaning_updates)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_config(path, project_root=tmp_path)


def test_real_config_has_approved_cleaning_rules():
    cleaning = load_config().cleaning
    assert set(cleaning.non_product_stock_codes) == {
        "POST", "DOT", "M", "m", "C2", "D", "S", "BANK CHARGES",
        "ADJUST", "ADJUST2", "AMAZONFEE", "CRUK", "B", "TEST001", "TEST002",
    }
    assert len(cleaning.non_product_stock_codes) == 15
    assert set(cleaning.unidentifiable_stock_codes) == {"DCGSSGIRL", "DCGSSBOY"}
    assert cleaning.voucher_stock_code_prefixes == ("gift_0001_",)
    assert cleaning.duplicate_policy == "flag"
    assert cleaning.overlap_policy == "keep_earlier_sheet"
    assert cleaning.cancellation_policy == "return"
    assert cleaning.zero_price_policy == "exclude"
    # Codes you decided to keep as products must not be excluded by any list.
    excluded = set(cleaning.non_product_stock_codes) | set(
        cleaning.unidentifiable_stock_codes
    )
    assert not any(code.startswith("DCGS0") or code == "PADS" for code in excluded)


def test_optional_settings_use_documented_defaults(tmp_path):
    cleaning = _load(tmp_path, {}).cleaning
    assert cleaning.overlap_policy == "keep_earlier_sheet"
    assert cleaning.cancellation_policy == "return"
    assert cleaning.zero_price_policy == "exclude"
    assert cleaning.unidentifiable_stock_codes == ()
    assert cleaning.voucher_stock_code_prefixes == ()


@pytest.mark.parametrize(
    "key,value",
    [
        ("overlap_policy", "keep_all"),
        ("cancellation_policy", "exclude"),
        ("zero_price_policy", "keep"),
    ],
)
def test_alternative_policy_values_load(tmp_path, key, value):
    cleaning = _load(tmp_path, {key: value}).cleaning
    assert getattr(cleaning, key) == value


@pytest.mark.parametrize(
    "key,value",
    [
        ("overlap_policy", "merge"),
        ("cancellation_policy", "ignore"),
        ("zero_price_policy", "maybe"),
    ],
)
def test_invalid_policy_values_raise(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        _load(tmp_path, {key: value})


def test_code_listed_in_both_lists_raises(tmp_path):
    with pytest.raises(ConfigError, match="both"):
        _load(tmp_path, {"unidentifiable_stock_codes": ["POST"]})


def test_null_optional_lists_become_empty(tmp_path):
    cleaning = _load(
        tmp_path,
        {"unidentifiable_stock_codes": None, "voucher_stock_code_prefixes": None},
    ).cleaning
    assert cleaning.unidentifiable_stock_codes == ()
    assert cleaning.voucher_stock_code_prefixes == ()


def test_voucher_prefixes_must_be_a_list(tmp_path):
    with pytest.raises(ConfigError, match="voucher_stock_code_prefixes"):
        _load(tmp_path, {"voucher_stock_code_prefixes": "gift_"})


def test_duplicate_unidentifiable_codes_raise(tmp_path):
    with pytest.raises(ConfigError, match="duplicate"):
        _load(tmp_path, {"unidentifiable_stock_codes": ["X", "X"]})


def test_blank_voucher_prefix_raises(tmp_path):
    with pytest.raises(ConfigError, match="empty"):
        _load(tmp_path, {"voucher_stock_code_prefixes": [""]})


def test_non_product_code_starting_with_voucher_prefix_raises(tmp_path):
    with pytest.raises(ConfigError, match="voucher"):
        _load(
            tmp_path,
            {
                "non_product_stock_codes": ["POST", "gift_0001_10"],
                "voucher_stock_code_prefixes": ["gift_0001_"],
            },
        )


def test_unidentifiable_code_starting_with_voucher_prefix_raises(tmp_path):
    with pytest.raises(ConfigError, match="voucher"):
        _load(
            tmp_path,
            {
                "unidentifiable_stock_codes": ["gift_X"],
                "voucher_stock_code_prefixes": ["gift_"],
            },
        )


def test_non_conflicting_voucher_prefix_loads(tmp_path):
    cleaning = _load(tmp_path, {"voucher_stock_code_prefixes": ["gift_"]}).cleaning
    assert cleaning.voucher_stock_code_prefixes == ("gift_",)
    assert cleaning.non_product_stock_codes == ("POST",)