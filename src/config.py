"""Load and validate project settings from configs/config.yaml.

Local-only: reads one YAML file with yaml.safe_load (no code execution) and
refuses any configured path that points outside the project folder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

VALID_DUPLICATE_POLICIES: tuple[str, ...] = ("flag", "drop")
VALID_OVERLAP_POLICIES: tuple[str, ...] = ("keep_earlier_sheet", "keep_all")
VALID_CANCELLATION_POLICIES: tuple[str, ...] = ("return", "exclude")
VALID_ZERO_PRICE_POLICIES: tuple[str, ...] = ("exclude", "keep")

DEFAULT_OVERLAP_POLICY: str = "keep_earlier_sheet"
DEFAULT_CANCELLATION_POLICY: str = "return"
DEFAULT_ZERO_PRICE_POLICY: str = "exclude"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    sheet_names: tuple[str, ...] | None
    processed_dir: Path


@dataclass(frozen=True)
class SchemaConfig:
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class CleaningConfig:
    cancellation_invoice_prefix: str
    non_product_stock_codes: tuple[str, ...]
    duplicate_policy: str
    overlap_policy: str = DEFAULT_OVERLAP_POLICY
    cancellation_policy: str = DEFAULT_CANCELLATION_POLICY
    zero_price_policy: str = DEFAULT_ZERO_PRICE_POLICY
    unidentifiable_stock_codes: tuple[str, ...] = ()
    voucher_stock_code_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    schema: SchemaConfig
    cleaning: CleaningConfig


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Missing or invalid section '{name}' in the config file.")
    return section


def _require_key(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing setting '{section_name}.{key}' in the config file.")
    return section[key]


def _read_choice(
    section: dict[str, Any],
    section_name: str,
    key: str,
    valid: tuple[str, ...],
    default: str,
) -> str:
    """Read an optional setting that must be one of a fixed set of choices."""
    value = section.get(key, default)
    if value not in valid:
        raise ConfigError(
            f"'{section_name}.{key}' must be one of {list(valid)}, got {value!r}."
        )
    return value


def _as_string_tuple(
    value: Any, label: str, *, allow_empty: bool, coerce_ints: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"'{label}' must be a list.")
    items: list[str] = []
    for item in value:
        if isinstance(item, bool):
            raise ConfigError(f"'{label}' contains an invalid value: {item!r}.")
        if isinstance(item, str):
            text = item.strip()
        elif coerce_ints and isinstance(item, int):
            text = str(item)
        else:
            raise ConfigError(f"'{label}' contains an invalid value: {item!r}.")
        if not text:
            raise ConfigError(f"'{label}' contains an empty value.")
        items.append(text)
    if not items and not allow_empty:
        raise ConfigError(f"'{label}' must not be empty.")
    if len(set(items)) != len(items):
        raise ConfigError(f"'{label}' contains duplicate values.")
    return tuple(items)


def _optional_string_tuple(
    section: dict[str, Any], key: str, label: str, *, coerce_ints: bool = False
) -> tuple[str, ...]:
    """Read an optional list setting; a missing or null value means 'empty'."""
    value = section.get(key)
    if value is None:
        return ()
    return _as_string_tuple(value, label, allow_empty=True, coerce_ints=coerce_ints)


def _resolve_inside_project(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{label}' must be a non-empty path string.")
    root = project_root.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ConfigError(
            f"'{label}' must be a relative path inside the project folder: {value!r}."
        ) from None
    return candidate


def load_config(
    path: Path | str | None = None, project_root: Path | None = None
) -> AppConfig:
    """Read, validate and return the project configuration."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    root = Path(project_root) if project_root is not None else PROJECT_ROOT

    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path.name}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a mapping of settings.")

    data = _require_section(raw, "data")
    sheets_value = _require_key(data, "data", "sheet_names")
    sheet_names = (
        None
        if sheets_value is None
        else _as_string_tuple(sheets_value, "data.sheet_names", allow_empty=False)
    )
    data_config = DataConfig(
        raw_path=_resolve_inside_project(
            root, _require_key(data, "data", "raw_path"), "data.raw_path"
        ),
        sheet_names=sheet_names,
        processed_dir=_resolve_inside_project(
            root, _require_key(data, "data", "processed_dir"), "data.processed_dir"
        ),
    )

    schema = _require_section(raw, "schema")
    schema_config = SchemaConfig(
        required_columns=_as_string_tuple(
            _require_key(schema, "schema", "required_columns"),
            "schema.required_columns",
            allow_empty=False,
        )
    )

    cleaning = _require_section(raw, "cleaning")
    prefix = _require_key(cleaning, "cleaning", "cancellation_invoice_prefix")
    if not isinstance(prefix, str) or len(prefix) != 1:
        raise ConfigError(
            "'cleaning.cancellation_invoice_prefix' must be a single character."
        )
    policy = _require_key(cleaning, "cleaning", "duplicate_policy")
    if policy not in VALID_DUPLICATE_POLICIES:
        raise ConfigError(
            "'cleaning.duplicate_policy' must be one of "
            f"{list(VALID_DUPLICATE_POLICIES)}, got {policy!r}."
        )

    non_product = _as_string_tuple(
        _require_key(cleaning, "cleaning", "non_product_stock_codes"),
        "cleaning.non_product_stock_codes",
        allow_empty=True,
        coerce_ints=True,
    )
    unidentifiable = _optional_string_tuple(
        cleaning,
        "unidentifiable_stock_codes",
        "cleaning.unidentifiable_stock_codes",
        coerce_ints=True,
    )
    in_both = sorted(set(non_product) & set(unidentifiable))
    if in_both:
        raise ConfigError(
            f"Stock code(s) {in_both} appear in both "
            "'cleaning.non_product_stock_codes' and "
            "'cleaning.unidentifiable_stock_codes'."
        )

    voucher_prefixes = _optional_string_tuple(
        cleaning,
        "voucher_stock_code_prefixes",
        "cleaning.voucher_stock_code_prefixes",
    )
    ambiguous = sorted(
        code
        for code in (*non_product, *unidentifiable)
        if any(code.startswith(voucher_prefix) for voucher_prefix in voucher_prefixes)
    )
    if ambiguous:
        raise ConfigError(
            f"Stock code(s) {ambiguous} are listed as excluded codes but also start "
            "with a prefix in 'cleaning.voucher_stock_code_prefixes'. "
            "A code may be one or the other, not both."
        )

    cleaning_config = CleaningConfig(
        cancellation_invoice_prefix=prefix,
        non_product_stock_codes=non_product,
        duplicate_policy=policy,
        overlap_policy=_read_choice(
            cleaning, "cleaning", "overlap_policy",
            VALID_OVERLAP_POLICIES, DEFAULT_OVERLAP_POLICY,
        ),
        cancellation_policy=_read_choice(
            cleaning, "cleaning", "cancellation_policy",
            VALID_CANCELLATION_POLICIES, DEFAULT_CANCELLATION_POLICY,
        ),
        zero_price_policy=_read_choice(
            cleaning, "cleaning", "zero_price_policy",
            VALID_ZERO_PRICE_POLICIES, DEFAULT_ZERO_PRICE_POLICY,
        ),
        unidentifiable_stock_codes=unidentifiable,
        voucher_stock_code_prefixes=voucher_prefixes,
    )

    return AppConfig(data=data_config, schema=schema_config, cleaning=cleaning_config)