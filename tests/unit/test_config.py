"""Tests for src.config.

All configuration data here is SYNTHETIC and written to temporary files.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config

VALID = {
    "data": {
        "raw_path": "data/raw/sample.xlsx",
        "sheet_names": None,
        "processed_dir": "data/processed",
    },
    "schema": {"required_columns": ["Invoice", "Price"]},
    "cleaning": {
        "cancellation_invoice_prefix": "C",
        "non_product_stock_codes": ["POST", 15],
        "duplicate_policy": "flag",
    },
}


def _valid() -> dict:
    return copy.deepcopy(VALID)


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, _valid()), project_root=tmp_path)
    assert cfg.data.raw_path == (tmp_path / "data" / "raw" / "sample.xlsx").resolve()
    assert cfg.data.processed_dir == (tmp_path / "data" / "processed").resolve()
    assert cfg.data.sheet_names is None
    assert cfg.schema.required_columns == ("Invoice", "Price")
    assert cfg.cleaning.cancellation_invoice_prefix == "C"
    assert cfg.cleaning.non_product_stock_codes == ("POST", "15")
    assert cfg.cleaning.duplicate_policy == "flag"


def test_real_project_config_loads():
    cfg = load_config()
    assert len(cfg.schema.required_columns) == 8
    assert "Customer ID" in cfg.schema.required_columns


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", project_root=tmp_path)


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path, project_root=tmp_path)


def test_non_mapping_file_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, project_root=tmp_path)


@pytest.mark.parametrize("section", ["data", "schema", "cleaning"])
def test_missing_section_raises(tmp_path, section):
    config = _valid()
    del config[section]
    with pytest.raises(ConfigError, match=section):
        load_config(_write(tmp_path, config), project_root=tmp_path)


@pytest.mark.parametrize(
    "section,key",
    [
        ("data", "raw_path"),
        ("data", "sheet_names"),
        ("data", "processed_dir"),
        ("schema", "required_columns"),
        ("cleaning", "cancellation_invoice_prefix"),
        ("cleaning", "non_product_stock_codes"),
        ("cleaning", "duplicate_policy"),
    ],
)
def test_missing_key_raises(tmp_path, section, key):
    config = _valid()
    del config[section][key]
    with pytest.raises(ConfigError, match=key):
        load_config(_write(tmp_path, config), project_root=tmp_path)


def test_invalid_duplicate_policy_raises(tmp_path):
    config = _valid()
    config["cleaning"]["duplicate_policy"] = "keep"
    with pytest.raises(ConfigError, match="duplicate_policy"):
        load_config(_write(tmp_path, config), project_root=tmp_path)


@pytest.mark.parametrize("prefix", ["CC", ""])
def test_invalid_cancellation_prefix_raises(tmp_path, prefix):
    config = _valid()
    config["cleaning"]["cancellation_invoice_prefix"] = prefix
    with pytest.raises(ConfigError, match="cancellation_invoice_prefix"):
        load_config(_write(tmp_path, config), project_root=tmp_path)


def test_relative_path_escaping_project_raises(tmp_path):
    config = _valid()
    config["data"]["raw_path"] = "../outside.xlsx"
    with pytest.raises(ConfigError, match="inside the project folder"):
        load_config(_write(tmp_path, config), project_root=tmp_path)


def test_absolute_path_outside_project_raises(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = _valid()
    config["data"]["raw_path"] = str(tmp_path / "elsewhere.xlsx")
    with pytest.raises(ConfigError, match="inside the project folder"):
        load_config(_write(tmp_path, config), project_root=project)


@pytest.mark.parametrize("columns", [[], ["A", "A"], "Invoice"])
def test_invalid_required_columns_raise(tmp_path, columns):
    config = _valid()
    config["schema"]["required_columns"] = columns
    with pytest.raises(ConfigError, match="required_columns"):
        load_config(_write(tmp_path, config), project_root=tmp_path)


def test_sheet_names_list_loads(tmp_path):
    config = _valid()
    config["data"]["sheet_names"] = ["Sheet1", "Sheet2"]
    cfg = load_config(_write(tmp_path, config), project_root=tmp_path)
    assert cfg.data.sheet_names == ("Sheet1", "Sheet2")


def test_empty_sheet_names_list_raises(tmp_path):
    config = _valid()
    config["data"]["sheet_names"] = []
    with pytest.raises(ConfigError, match="sheet_names"):
        load_config(_write(tmp_path, config), project_root=tmp_path)


def test_boolean_stock_code_raises(tmp_path):
    config = _valid()
    config["cleaning"]["non_product_stock_codes"] = [True]
    with pytest.raises(ConfigError, match="non_product_stock_codes"):
        load_config(_write(tmp_path, config), project_root=tmp_path)