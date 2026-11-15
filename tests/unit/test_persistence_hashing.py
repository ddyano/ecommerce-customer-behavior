"""Tests for src.persistence.hashing.

All data here is SYNTHETIC (invented values). Nothing comes from the real dataset.
Two groups of tests read the real project: the config tests read the real
configs/config.yaml, and the code-fingerprint tests read the real source files
(to check that the fingerprinted list matches the pipeline's import closure).
"""
from __future__ import annotations

import ast
import dataclasses
import datetime
import hashlib
import random
import struct
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from src.config import PROJECT_ROOT, CleaningConfig, load_config
from src.persistence.hashing import (
    DEFAULT_CODE_FILES,
    FINGERPRINT_VERSION,
    ConfigHashError,
    FingerprintError,
    HashingError,
    canonical_json_bytes,
    differing_columns,
    fingerprint_code,
    fingerprint_dataframe,
    hash_cleaning_config,
    hash_file,
    sha256_hex,
)


def lp(data: bytes) -> bytes:
    return struct.pack("<Q", len(data)) + data


def fp(frame, **kwargs):
    return fingerprint_dataframe(frame, **kwargs)


def single(series: pd.Series) -> str:
    """Digest of a one-column frame, for comparing individual columns."""
    return fp(pd.DataFrame({"c": series})).digest


def obj(*values) -> pd.Series:
    return pd.Series(list(values), dtype=object)


# ============================== hash_file ====================================
def test_hash_file_known_content(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    result = hash_file(path)
    assert result.name == "f.bin"
    assert result.size_bytes == 3
    assert result.sha256 == hashlib.sha256(b"abc").hexdigest()


@pytest.mark.parametrize("chunk_bytes", [1, 7, 4096, 1024 * 1024])
def test_hash_file_chunking_does_not_change_digest(tmp_path, chunk_bytes):
    data = bytes(range(256)) * 300
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    result = hash_file(path, chunk_bytes=chunk_bytes)
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    assert result.size_bytes == len(data)


def test_hash_file_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    result = hash_file(path)
    assert result.size_bytes == 0
    assert result.sha256 == hashlib.sha256(b"").hexdigest()


def test_hash_file_missing_file_raises(tmp_path):
    with pytest.raises(HashingError, match="nope.bin") as excinfo:
        hash_file(tmp_path / "nope.bin")
    assert str(tmp_path) not in str(excinfo.value)


def test_hash_file_directory_raises(tmp_path):
    with pytest.raises(HashingError, match=tmp_path.name):
        hash_file(tmp_path)


@pytest.mark.parametrize("bad", [0, -5, 1.5, True])
def test_hash_file_invalid_chunk_size_raises(tmp_path, bad):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError):
        hash_file(path, chunk_bytes=bad)


# ========================= digests and canonical JSON ========================
def test_sha256_hex_known_value():
    assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_canonical_json_ignores_key_order():
    assert canonical_json_bytes({"b": 1, "a": [2, 3]}) == b'{"a":[2,3],"b":1}'
    assert canonical_json_bytes({"a": [2, 3], "b": 1}) == b'{"a":[2,3],"b":1}'


def test_canonical_json_is_compact_and_ascii():
    assert canonical_json_bytes({"\u00e9": "\u00fc"}) == b'{"\\u00e9":"\\u00fc"}'


def test_canonical_json_treats_tuples_like_lists():
    assert canonical_json_bytes({"a": (1, 2)}) == canonical_json_bytes({"a": [1, 2]})


def test_canonical_json_rejects_nan():
    with pytest.raises(HashingError):
        canonical_json_bytes({"a": float("nan")})


def test_canonical_json_rejects_unsupported_types():
    with pytest.raises(HashingError):
        canonical_json_bytes({"a": {1, 2}})


# ============================ cleaning config hash ===========================
def base_cleaning() -> CleaningConfig:
    return CleaningConfig(
        cancellation_invoice_prefix="C",
        non_product_stock_codes=("POST", "M"),
        duplicate_policy="flag",
        overlap_policy="keep_earlier_sheet",
        cancellation_policy="return",
        zero_price_policy="exclude",
        unidentifiable_stock_codes=("DCGSSGIRL", "DCGSSBOY"),
        voucher_stock_code_prefixes=("gift_0001_", "vch_"),
    )


def config_digest(cleaning=None, required=("Invoice", "Price"), sheets=None) -> str:
    cleaning = base_cleaning() if cleaning is None else cleaning
    return hash_cleaning_config(cleaning, list(required), sheets).digest


def test_same_config_gives_same_digest():
    first = hash_cleaning_config(base_cleaning(), ["Invoice", "Price"], None)
    second = hash_cleaning_config(base_cleaning(), ["Invoice", "Price"], None)
    assert first == second
    assert len(first.digest) == 64
    assert first.version == 1


def test_canonical_form_covers_every_dataclass_field():
    result = hash_cleaning_config(base_cleaning(), ["Invoice", "Price"], None)
    assert set(result.canonical) == {
        "config_hash_version",
        "cleaning",
        "required_columns",
        "sheet_names",
    }
    assert set(result.canonical["cleaning"]) == {
        f.name for f in dataclasses.fields(CleaningConfig)
    }


def test_config_hash_canonical_form_is_pinned():
    """If this fails the canonical form changed: bump CONFIG_HASH_VERSION as well."""
    result = hash_cleaning_config(base_cleaning(), ["Invoice", "Price"], None)
    assert result.version == 1
    assert result.canonical == {
        "config_hash_version": 1,
        "cleaning": {
            "cancellation_invoice_prefix": "C",
            "non_product_stock_codes": ["M", "POST"],
            "duplicate_policy": "flag",
            "overlap_policy": "keep_earlier_sheet",
            "cancellation_policy": "return",
            "zero_price_policy": "exclude",
            "unidentifiable_stock_codes": ["DCGSSBOY", "DCGSSGIRL"],
            "voucher_stock_code_prefixes": ["gift_0001_", "vch_"],
        },
        "required_columns": ["Invoice", "Price"],
        "sheet_names": None,
    }
    assert result.digest == hashlib.sha256(canonical_json_bytes(result.canonical)).hexdigest()


def test_set_like_lists_ignore_order():
    reordered = replace(
        base_cleaning(),
        non_product_stock_codes=("M", "POST"),
        unidentifiable_stock_codes=("DCGSSBOY", "DCGSSGIRL"),
        voucher_stock_code_prefixes=("vch_", "gift_0001_"),
    )
    assert config_digest(reordered) == config_digest()


def test_required_columns_order_is_preserved():
    original = hash_cleaning_config(base_cleaning(), ["Invoice", "Price"], None)
    reordered = hash_cleaning_config(base_cleaning(), ["Price", "Invoice"], None)
    assert reordered.digest != original.digest
    assert original.canonical["required_columns"] == ["Invoice", "Price"]
    assert reordered.canonical["required_columns"] == ["Price", "Invoice"]


def test_sheet_names_order_matters():
    assert config_digest(sheets=("A", "B")) != config_digest(sheets=("B", "A"))


def test_sheet_names_none_differs_from_a_list():
    assert config_digest(sheets=None) != config_digest(sheets=("A",))


FIELD_CHANGES = [
    ("cancellation_invoice_prefix", "X"),
    ("non_product_stock_codes", ("POST", "M", "NEW")),
    ("duplicate_policy", "drop"),
    ("overlap_policy", "keep_all"),
    ("cancellation_policy", "exclude"),
    ("zero_price_policy", "keep"),
    ("unidentifiable_stock_codes", ("DCGSSGIRL",)),
    ("voucher_stock_code_prefixes", ("gift_0001_",)),
]


@pytest.mark.parametrize("name,value", FIELD_CHANGES)
def test_every_cleaning_field_change_alters_the_digest(name, value):
    changed = replace(base_cleaning(), **{name: value})
    assert config_digest(changed) != config_digest()


def test_required_columns_change_alters_the_digest():
    assert config_digest(required=("Invoice", "Price", "Extra")) != config_digest()


def test_future_dataclass_field_is_included():
    @dataclass(frozen=True)
    class FutureConfig:
        cancellation_invoice_prefix: str = "C"
        new_option: str = "a"
        ordered_steps: tuple = ("a", "b")

    base = hash_cleaning_config(FutureConfig(), ["a"], None)
    assert base.canonical["cleaning"]["new_option"] == "a"
    assert base.canonical["cleaning"]["ordered_steps"] == ["a", "b"]
    changed = hash_cleaning_config(FutureConfig(new_option="b"), ["a"], None)
    assert changed.digest != base.digest
    # A field that is not known to be set-like keeps its order.
    reordered = hash_cleaning_config(FutureConfig(ordered_steps=("b", "a")), ["a"], None)
    assert reordered.digest != base.digest


def test_unsupported_field_type_raises():
    @dataclass(frozen=True)
    class OddConfig:
        options: dict = field(default_factory=dict)

    with pytest.raises(ConfigHashError, match="options"):
        hash_cleaning_config(OddConfig(), ["a"], None)


@pytest.mark.parametrize("bad", [{"a": 1}, CleaningConfig], ids=["dict", "class"])
def test_non_dataclass_input_raises(bad):
    with pytest.raises(ConfigHashError, match="dataclass instance"):
        hash_cleaning_config(bad, ["a"], None)


def test_non_string_set_like_item_raises():
    bad = replace(base_cleaning(), non_product_stock_codes=(1, 2))
    with pytest.raises(ConfigHashError, match="non_product_stock_codes"):
        hash_cleaning_config(bad, ["a"], None)


@pytest.mark.parametrize("bad", ["Invoice", ["a", 1], None], ids=["string", "mixed", "none"])
def test_required_columns_must_be_a_list_of_strings(bad):
    with pytest.raises(ConfigHashError, match="required_columns"):
        hash_cleaning_config(base_cleaning(), bad, None)


def test_non_finite_float_field_raises():
    @dataclass(frozen=True)
    class FloatConfig:
        threshold: float = float("nan")

    with pytest.raises(ConfigHashError, match="threshold"):
        hash_cleaning_config(FloatConfig(), ["a"], None)


def test_real_config_hash_is_stable_and_sensitive():
    cfg = load_config()
    first = hash_cleaning_config(cfg.cleaning, cfg.schema.required_columns, cfg.data.sheet_names)
    second = hash_cleaning_config(cfg.cleaning, cfg.schema.required_columns, cfg.data.sheet_names)
    assert first.digest == second.digest
    smaller = replace(
        cfg.cleaning, non_product_stock_codes=cfg.cleaning.non_product_stock_codes[:-1]
    )
    changed = hash_cleaning_config(smaller, cfg.schema.required_columns, cfg.data.sheet_names)
    assert changed.digest != first.digest


YAML_TEMPLATE = """{comment}
data:
  raw_path: {raw}
  sheet_names: null
  processed_dir: data/processed
schema:
  required_columns:
    - Invoice
    - Price
cleaning:
  cancellation_invoice_prefix: "C"
  duplicate_policy: flag
  non_product_stock_codes:
    - "POST"
"""


def test_paths_and_comments_do_not_affect_the_digest(tmp_path):
    first_path = tmp_path / "a.yaml"
    second_path = tmp_path / "b.yaml"
    first_path.write_text(
        YAML_TEMPLATE.format(comment="", raw="data/raw/a.xlsx"), encoding="utf-8"
    )
    second_path.write_text(
        YAML_TEMPLATE.format(comment="# only a comment", raw="elsewhere/b.xlsx"),
        encoding="utf-8",
    )
    digests = []
    for path in (first_path, second_path):
        cfg = load_config(path, project_root=tmp_path)
        digests.append(
            hash_cleaning_config(
                cfg.cleaning, cfg.schema.required_columns, cfg.data.sheet_names
            ).digest
        )
    assert digests[0] == digests[1]


# ============================== code fingerprint =============================
def make_project(root, files: dict):
    root.mkdir(parents=True, exist_ok=True)
    for relative, data in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def test_per_file_hashes_use_normalised_content(tmp_path):
    root = make_project(tmp_path / "p", {"a.py": b"x = 1\n", "pkg/b.py": b"y = 2\n"})
    result = fingerprint_code(root, ("pkg/b.py", "a.py"))
    assert result.files == (
        ("a.py", hashlib.sha256(b"x = 1\n").hexdigest()),
        ("pkg/b.py", hashlib.sha256(b"y = 2\n").hexdigest()),
    )
    assert len(result.digest) == 64
    assert result.version == 1


def test_code_fingerprint_digest_format_is_pinned(tmp_path):
    """If this fails the digest format changed: bump CODE_FINGERPRINT_VERSION as well."""
    root = make_project(tmp_path / "p", {"a.py": b"x = 1\n", "pkg/b.py": b"y = 2\n"})
    result = fingerprint_code(root, ("pkg/b.py", "a.py"))
    expected = hashlib.sha256(
        canonical_json_bytes(
            {
                "code_fingerprint_version": 1,
                "files": [
                    ["a.py", hashlib.sha256(b"x = 1\n").hexdigest()],
                    ["pkg/b.py", hashlib.sha256(b"y = 2\n").hexdigest()],
                ],
            }
        )
    ).hexdigest()
    assert result.version == 1
    assert result.digest == expected


def test_line_endings_do_not_change_the_fingerprint(tmp_path):
    crlf = make_project(tmp_path / "crlf", {"a.py": b"x = 1\r\ny = 2\r\n"})
    lf = make_project(tmp_path / "lf", {"a.py": b"x = 1\ny = 2\n"})
    assert fingerprint_code(crlf, ("a.py",)) == fingerprint_code(lf, ("a.py",))


def test_changing_one_file_changes_only_its_hash(tmp_path):
    first = make_project(tmp_path / "one", {"a.py": b"x = 1\n", "b.py": b"y = 2\n"})
    second = make_project(tmp_path / "two", {"a.py": b"x = 1\n", "b.py": b"y = 3\n"})
    before = fingerprint_code(first, ("a.py", "b.py"))
    after = fingerprint_code(second, ("a.py", "b.py"))
    assert before.digest != after.digest
    assert before.files[0] == after.files[0]
    assert before.files[1] != after.files[1]


def test_input_order_does_not_matter(tmp_path):
    root = make_project(tmp_path / "p", {"a.py": b"x = 1\n", "b.py": b"y = 2\n"})
    assert fingerprint_code(root, ("a.py", "b.py")) == fingerprint_code(root, ("b.py", "a.py"))


def test_missing_file_raises(tmp_path):
    root = make_project(tmp_path / "p", {"a.py": b"x = 1\n"})
    with pytest.raises(HashingError, match="missing.py"):
        fingerprint_code(root, ("a.py", "missing.py"))


@pytest.mark.parametrize("kind", ["parent_relative", "absolute"])
def test_path_outside_project_raises(tmp_path, kind):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"z = 3\n")
    path = "../outside.py" if kind == "parent_relative" else str(outside)
    with pytest.raises(HashingError, match="outside"):
        fingerprint_code(root, (path,))


def test_duplicate_paths_raise(tmp_path):
    root = make_project(tmp_path / "p", {"a.py": b"x = 1\n"})
    with pytest.raises(HashingError, match="Duplicate"):
        fingerprint_code(root, ("a.py", "./a.py"))


def test_empty_file_list_raises(tmp_path):
    with pytest.raises(HashingError, match="No source files"):
        fingerprint_code(tmp_path, ())


def test_windows_separators_are_accepted(tmp_path):
    root = make_project(tmp_path / "p", {"pkg/b.py": b"y = 2\n"})
    result = fingerprint_code(root, ("pkg\\b.py",))
    assert result.files[0][0] == "pkg/b.py"


def test_default_files_exist_in_the_real_project():
    first = fingerprint_code(PROJECT_ROOT)
    second = fingerprint_code(PROJECT_ROOT)
    assert first == second
    assert [path for path, _ in first.files] == sorted(DEFAULT_CODE_FILES)
    assert len(first.files) == 6
    assert all(len(file_hash) == 64 for _, file_hash in first.files)


def test_default_code_files_include_the_config_module():
    assert "src/config.py" in DEFAULT_CODE_FILES
    assert len(set(DEFAULT_CODE_FILES)) == len(DEFAULT_CODE_FILES)


# ----- import-closure check: is the fingerprinted list complete and minimal? -----
PIPELINE_ENTRY_FILES = ("src/ingestion/loader.py", "src/preprocessing/cleaning.py")


def _resolve_first_party(root: Path, dotted: str) -> str | None:
    """Relative path of the src module a dotted name refers to, if it is a module file."""
    if dotted != "src" and not dotted.startswith("src."):
        return None
    candidate = root.joinpath(*dotted.split(".")).with_suffix(".py")
    return candidate.relative_to(root).as_posix() if candidate.is_file() else None


def _first_party_imports(root: Path, relative_path: str) -> set[str]:
    source = (root / relative_path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=relative_path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError(
                    f"{relative_path}: relative imports are not supported by this check"
                )
            base = node.module or ""
            names = [base] + [f"{base}.{alias.name}" for alias in node.names]
        else:
            continue
        for dotted in names:
            resolved = _resolve_first_party(root, dotted)
            if resolved is not None:
                found.add(resolved)
    return found


def import_closure(root: Path, entry_files) -> set[str]:
    """Every first-party src module file reachable from the entry files."""
    closure: set[str] = set()
    pending = list(entry_files)
    while pending:
        current = pending.pop()
        if current in closure:
            continue
        closure.add(current)
        pending.extend(_first_party_imports(root, current) - closure)
    return closure


def init_files_on_import_paths(root: Path, relative_files) -> list[str]:
    """The existing __init__.py files of every package the given modules are imported through."""
    found: set[str] = set()
    for relative in relative_files:
        parts = Path(relative).parts[:-1]
        for depth in range(1, len(parts) + 1):
            candidate = root.joinpath(*parts[:depth], "__init__.py")
            if candidate.is_file():
                found.add(candidate.relative_to(root).as_posix())
    return sorted(found)


def init_files_with_code(root: Path, relative_inits) -> list[str]:
    """Of the given __init__.py files, those containing more than a docstring or pass."""
    offenders = []
    for relative in relative_inits:
        body = ast.parse((root / relative).read_text(encoding="utf-8-sig")).body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if any(not isinstance(node, ast.Pass) for node in body):
            offenders.append(relative)
    return sorted(offenders)


def dynamic_import_calls(root: Path, relative_path: str) -> list[str]:
    """Calls that import or run code by name at runtime (invisible to a static scan).

    Flags import_module(...) (plain or as a method), and the builtins __import__,
    exec and eval. Methods that merely share a name (for example DataFrame.eval) are
    not flagged. This is a guard, not a proof: names built from strings in other
    ways (getattr tricks, plugin loaders) cannot be detected.
    """
    source = (root / relative_path).read_text(encoding="utf-8-sig")
    found = []
    for node in ast.walk(ast.parse(source, filename=relative_path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            flagged = name in {"__import__", "import_module", "exec", "eval"}
        elif isinstance(func, ast.Attribute):
            name = func.attr
            flagged = name == "import_module"
        else:
            continue
        if flagged:
            found.append(f"{relative_path}:{node.lineno}: {name}(...)")
    return sorted(found)


def write_modules(root: Path, files: dict) -> Path:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def test_scanner_follows_every_import_form(tmp_path):
    root = write_modules(
        tmp_path / "p",
        {
            "src/a.py": (
                "import json\nimport pandas as pd\nimport src.b\n"
                "from src.c import thing\nfrom src import d\n"
                "from src.pkg import mod, CONSTANT\n"
                "def later():\n    from src.late import value\n"
            ),
            "src/b.py": "",
            "src/c.py": "thing = 1\n",
            "src/d.py": "",
            "src/late.py": "value = 1\n",
            "src/pkg/__init__.py": "CONSTANT = 1\n",
            "src/pkg/mod.py": "",
            "src/unused.py": "",
        },
    )
    assert import_closure(root, ("src/a.py",)) == {
        "src/a.py",
        "src/b.py",
        "src/c.py",
        "src/d.py",
        "src/late.py",
        "src/pkg/mod.py",
    }


def test_scanner_is_transitive_and_handles_cycles(tmp_path):
    root = write_modules(
        tmp_path / "p",
        {
            "src/a.py": "import src.b\n",
            "src/b.py": "import src.c\n",
            "src/c.py": "import src.a\nimport src.d\n",
            "src/d.py": "",
        },
    )
    assert import_closure(root, ("src/a.py",)) == {
        "src/a.py",
        "src/b.py",
        "src/c.py",
        "src/d.py",
    }


def test_scanner_ignores_modules_outside_src_and_unresolvable_names(tmp_path):
    root = write_modules(
        tmp_path / "p",
        {
            "src/a.py": "import scripts.tool\nimport src.missing\nfrom src.b import nothing\n",
            "src/b.py": "",
            "scripts/tool.py": "",
        },
    )
    assert import_closure(root, ("src/a.py",)) == {"src/a.py", "src/b.py"}


def test_scanner_rejects_relative_imports(tmp_path):
    root = write_modules(
        tmp_path / "p", {"src/a.py": "from . import b\n", "src/b.py": ""}
    )
    with pytest.raises(ValueError, match="relative"):
        import_closure(root, ("src/a.py",))


def test_scanner_missing_entry_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_closure(tmp_path, ("src/nope.py",))


def test_scanner_accepts_a_utf8_byte_order_mark(tmp_path):
    root = tmp_path / "p"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_bytes(b"\xef\xbb\xbfimport src.b\n")
    (root / "src" / "b.py").write_bytes(b"")
    assert import_closure(root, ("src/a.py",)) == {"src/a.py", "src/b.py"}


def test_init_files_on_import_paths_lists_only_packages_that_are_imported(tmp_path):
    root = write_modules(
        tmp_path / "p",
        {
            "src/__init__.py": "",
            "src/a/__init__.py": "",
            "src/a/b/__init__.py": "",
            "src/a/b/mod.py": "",
            "src/other/__init__.py": "VALUE = 1\n",
            "src/other/thing.py": "",
            "src/top.py": "",
        },
    )
    assert init_files_on_import_paths(root, ["src/a/b/mod.py", "src/top.py"]) == [
        "src/__init__.py",
        "src/a/__init__.py",
        "src/a/b/__init__.py",
    ]


def test_init_files_on_import_paths_tolerates_packages_without_init_files(tmp_path):
    root = write_modules(tmp_path / "p", {"src/__init__.py": "", "src/ns/mod.py": ""})
    assert init_files_on_import_paths(root, ["src/ns/mod.py"]) == ["src/__init__.py"]


def test_init_file_check_flags_only_files_with_code(tmp_path):
    root = write_modules(
        tmp_path / "p",
        {
            "src/__init__.py": "",
            "src/a/__init__.py": "# only a comment\n",
            "src/b/__init__.py": '"""Docstring only."""\n',
            "src/c/__init__.py": "pass\n",
            "src/d/__init__.py": "VALUE = 1\n",
            "src/e/__init__.py": "from src.b import thing\n",
        },
    )
    inits = [
        "src/__init__.py",
        "src/a/__init__.py",
        "src/b/__init__.py",
        "src/c/__init__.py",
        "src/d/__init__.py",
        "src/e/__init__.py",
    ]
    assert init_files_with_code(root, inits) == ["src/d/__init__.py", "src/e/__init__.py"]


def test_dynamic_import_detector_flags_runtime_imports(tmp_path):
    root = write_modules(
        tmp_path / "p",
        {
            "src/a.py": (
                "import importlib\n"
                "from importlib import import_module\n"
                "def f(name):\n"
                "    importlib.import_module(name)\n"
                "    import_module(name)\n"
                "    __import__(name)\n"
                "    exec('x = 1')\n"
                "    eval('1')\n"
            )
        },
    )
    assert dynamic_import_calls(root, "src/a.py") == [
        "src/a.py:4: import_module(...)",
        "src/a.py:5: import_module(...)",
        "src/a.py:6: __import__(...)",
        "src/a.py:7: exec(...)",
        "src/a.py:8: eval(...)",
    ]


def test_dynamic_import_detector_ignores_methods_with_similar_names(tmp_path):
    root = write_modules(
        tmp_path / "p",
        {"src/a.py": "def f(df, obj):\n    df.eval('a + b')\n    obj.exec()\n    obj.__import__()\n"},
    )
    assert dynamic_import_calls(root, "src/a.py") == []


def test_default_code_files_are_exactly_the_pipeline_import_closure():
    closure = import_closure(PROJECT_ROOT, PIPELINE_ENTRY_FILES)
    listed = set(DEFAULT_CODE_FILES)
    assert listed == closure, (
        f"missing from DEFAULT_CODE_FILES: {sorted(closure - listed)}; "
        f"listed but not imported by the pipeline: {sorted(listed - closure)}"
    )


def test_package_init_files_on_the_pipeline_import_path_contain_no_code():
    closure = import_closure(PROJECT_ROOT, PIPELINE_ENTRY_FILES)
    inits = init_files_on_import_paths(PROJECT_ROOT, closure)
    assert init_files_with_code(PROJECT_ROOT, inits) == []


def test_pipeline_files_use_no_dynamic_imports():
    closure = import_closure(PROJECT_ROOT, PIPELINE_ENTRY_FILES)
    found = [
        call for relative in sorted(closure) for call in dynamic_import_calls(PROJECT_ROOT, relative)
    ]
    assert found == [], f"runtime imports hide modules from the closure check: {found}"


# ============================ DataFrame fingerprint ==========================
def test_structure_and_determinism():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = fp(frame)
    assert result.version == FINGERPRINT_VERSION == 1
    assert len(result.digest) == 64
    assert result.rows == 2
    assert [name for name, _ in result.columns] == ["a", "b"]
    assert result.columns[0][1] == "int64"
    assert result.columns[1][1] == "values"
    assert [name for name, _ in result.column_digests] == ["a", "b"]
    assert fp(frame) == result


NAN_PAYLOADS = np.array(
    [0x7FF8000000000001, 0xFFF8000000000000], dtype=np.uint64
).view(np.float64)
US_SERIES = pd.Series(pd.to_datetime(["2020-01-01 10:00"])).astype("datetime64[us]")
NS_SERIES = pd.Series(pd.to_datetime(["2020-01-01 10:00"])).astype("datetime64[ns]")
NAT_SERIES = pd.Series([pd.NaT], dtype="datetime64[us]")
STAMP_SERIES = pd.Series(pd.to_datetime(["2020-01-01"])).astype("datetime64[us]")
NEXT_AFTER_TENTH = float(np.nextafter(0.1, 1.0))

DISTINCT_PAIRS = [
    pytest.param(obj(5), obj("5"), id="int_vs_str"),
    pytest.param(obj(5.0), obj(5), id="float_vs_int"),
    pytest.param(obj(True), obj(1), id="bool_vs_int"),
    pytest.param(obj(False), obj(0), id="false_vs_zero"),
    pytest.param(obj(True), obj(False), id="true_vs_false"),
    pytest.param(obj(""), obj(np.nan), id="empty_string_vs_nan"),
    pytest.param(obj(""), obj(None), id="empty_string_vs_none"),
    pytest.param(obj("nan"), obj(np.nan), id="text_nan_vs_nan"),
    pytest.param(obj("None"), obj(None), id="text_none_vs_none"),
    pytest.param(obj(np.nan), obj(None), id="nan_vs_none"),
    pytest.param(obj(np.nan), obj(pd.NA), id="nan_vs_na"),
    pytest.param(obj(None), obj(pd.NA), id="none_vs_na"),
    pytest.param(obj(0.0), obj(-0.0), id="object_zero_vs_negative_zero"),
    pytest.param(obj(0.1), obj(NEXT_AFTER_TENTH), id="adjacent_floats"),
    pytest.param(obj(float("inf")), obj(float("-inf")), id="inf_vs_negative_inf"),
    pytest.param(obj(2**70), obj(2**70 + 1), id="large_ints"),
    pytest.param(obj(-1), obj(1), id="negative_vs_positive"),
    pytest.param(obj("\u00e9"), obj("e\u0301"), id="no_unicode_normalisation"),
    pytest.param(obj("a"), obj("a "), id="trailing_space"),
    pytest.param(obj("a"), obj("A"), id="case"),
    pytest.param(obj(b"x"), obj("x"), id="bytes_vs_str"),
    pytest.param(obj(np.nan), pd.Series([np.nan], dtype="float64"), id="object_nan_vs_float_column"),
    pytest.param(pd.Series([0.0]), pd.Series([-0.0]), id="float_zero_vs_negative_zero"),
    pytest.param(pd.Series([5.0]), pd.Series([5], dtype="int64"), id="float_column_vs_int_column"),
    pytest.param(pd.Series([True]), pd.Series([1], dtype="int64"), id="bool_column_vs_int_column"),
    pytest.param(pd.Series([1], dtype="int64"), pd.Series([1], dtype="int32"), id="int64_vs_int32"),
    pytest.param(pd.Series([1], dtype="uint8"), pd.Series([1], dtype="int8"), id="uint8_vs_int8"),
    pytest.param(pd.Series([1.5], dtype="float32"), pd.Series([1.5], dtype="float64"), id="float32_vs_float64"),
    pytest.param(US_SERIES, NS_SERIES, id="datetime_us_vs_ns"),
    pytest.param(NAT_SERIES, STAMP_SERIES, id="nat_vs_timestamp"),
]


@pytest.mark.parametrize("left,right", DISTINCT_PAIRS)
def test_distinct_values_alter_the_fingerprint(left, right):
    assert single(left) != single(right)


TEXT_VALUES = ["a", None, "b", "a", ""]


def text_series(kind: str) -> pd.Series:
    """The SAME text column in each representation that can occur."""
    if kind == "object":
        return pd.Series(["a", np.nan, "b", "a", ""], dtype=object)
    if kind == "string_na":
        return pd.Series(TEXT_VALUES, dtype="string")
    if kind == "str_nan":
        return pd.Series(TEXT_VALUES, dtype=pd.StringDtype(na_value=np.nan))
    if kind == "inferred":
        return pd.Series(TEXT_VALUES)
    if kind == "categorical":
        return pd.Series(pd.Categorical(TEXT_VALUES))
    if kind == "categorical_on_str":
        return pd.Series(TEXT_VALUES, dtype=pd.StringDtype(na_value=np.nan)).astype("category")
    raise AssertionError(kind)


TEXT_KINDS = [
    "object",
    "string_na",
    "str_nan",
    "inferred",
    "categorical",
    "categorical_on_str",
]


@pytest.mark.parametrize("kind", TEXT_KINDS)
def test_one_text_column_has_the_same_fingerprint_in_every_representation(kind):
    assert single(text_series(kind)) == single(text_series("object"))


def test_text_representations_still_differ_when_values_differ():
    changed = pd.Series(["a", None, "b", "a", "x"], dtype=pd.StringDtype(na_value=np.nan))
    assert single(changed) != single(text_series("object"))


EQUAL_PAIRS = [
    pytest.param(pd.Series(["a", "b"]), obj("a", "b"), id="inferred_text_vs_object"),
    pytest.param(
        obj("a", np.nan),
        pd.Series(["a", None], dtype="string"),
        id="object_nan_vs_string_dtype_missing",
    ),
    pytest.param(
        obj("a", np.nan, "b"),
        pd.Series(pd.Categorical(["a", None, "b"])),
        id="object_nan_vs_categorical_missing",
    ),
    pytest.param(
        pd.Series(pd.Categorical(["a", "b", "a"], categories=["b", "a"])),
        obj("a", "b", "a"),
        id="category_order_ignored",
    ),
    pytest.param(
        pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b", "z"])),
        obj("a", "b", "a"),
        id="unused_categories_ignored",
    ),
    pytest.param(
        pd.Series(pd.Categorical([1, 2, 1])), obj(1, 2, 1), id="integer_categories_vs_object_ints"
    ),
    pytest.param(pd.Series(NAN_PAYLOADS), pd.Series([np.nan, np.nan]), id="nan_bit_patterns"),
    pytest.param(
        pd.Series(np.array(["NaT", "NaT"], dtype="datetime64[us]")),
        pd.Series([pd.NaT, pd.NaT], dtype="datetime64[us]"),
        id="nat_variants",
    ),
    pytest.param(
        pd.Series(["a", pd.NA], dtype="string"),
        pd.Series(pd.Categorical(["a", None])),
        id="string_dtype_missing_vs_categorical_missing",
    ),
    pytest.param(pd.Series(["x", "y"], dtype="string"), obj("x", "y"), id="string_dtype_vs_object"),
]


@pytest.mark.parametrize("left,right", EQUAL_PAIRS)
def test_equal_logical_values_give_equal_fingerprints(left, right):
    assert single(left) == single(right)


def test_value_tags_are_pinned():
    series = pd.Series(["5", 5, 5.0, True, None, pd.NA, b"x", np.nan], dtype=object)
    stream = (
        b"\x03" + lp(b"5")
        + b"\x04" + lp(b"5")
        + b"\x05" + lp(float.hex(5.0).encode("ascii"))
        + b"\x06\x01"
        + b"\x01"
        + b"\x02"
        + b"\x07" + lp(b"x")
        + b"\x00"
    )
    expected = hashlib.sha256(b"column-v1" + lp(b"values") + stream).hexdigest()
    assert dict(fp(pd.DataFrame({"c": series})).column_digests)["c"] == expected


def test_format_is_pinned():
    frame = pd.DataFrame(
        {
            "n": np.array([1, 2, 3], dtype="int64"),
            "t": pd.Series(["a", np.nan, "bc"], dtype=object),
        }
    )
    n_column = hashlib.sha256(
        b"column-v1" + lp(b"int64") + np.array([1, 2, 3], dtype="<i8").tobytes()
    ).digest()
    t_stream = b"\x03" + lp(b"a") + b"\x00" + b"\x03" + lp(b"bc")
    t_column = hashlib.sha256(b"column-v1" + lp(b"values") + t_stream).digest()
    expected = hashlib.sha256(
        b"dataframe-v1"
        + struct.pack("<I", 1)
        + struct.pack("<Q", 3)
        + struct.pack("<Q", 2)
        + lp(b"n") + lp(b"int64") + n_column
        + lp(b"t") + lp(b"values") + t_column
    ).hexdigest()
    result = fp(frame)
    assert result.digest == expected
    assert dict(result.column_digests) == {"n": n_column.hex(), "t": t_column.hex()}


def test_datetime_bytes_are_pinned():
    series = pd.Series(
        pd.to_datetime(["2020-01-01 00:00:00", None, "1970-01-01 00:00:01"])
    ).astype("datetime64[us]")
    raw = np.array(
        [1_577_836_800_000_000, np.iinfo(np.int64).min, 1_000_000], dtype="<i8"
    ).tobytes()
    expected = hashlib.sha256(b"column-v1" + lp(b"datetime64[us]") + raw).hexdigest()
    assert dict(fp(pd.DataFrame({"c": series})).column_digests)["c"] == expected


def test_float_bytes_are_pinned_with_canonical_nan_and_signed_zero():
    series = pd.Series(np.array([1.5, np.nan, -0.0, NAN_PAYLOADS[0]]), dtype="float64")
    raw = np.array([1.5, np.nan, -0.0, np.nan], dtype="<f8").tobytes()
    expected = hashlib.sha256(b"column-v1" + lp(b"float64") + raw).hexdigest()
    assert dict(fp(pd.DataFrame({"c": series})).column_digests)["c"] == expected


def test_bool_bytes_are_pinned():
    series = pd.Series([True, False, True])
    expected = hashlib.sha256(b"column-v1" + lp(b"bool") + bytes([1, 0, 1])).hexdigest()
    assert dict(fp(pd.DataFrame({"c": series})).column_digests)["c"] == expected


def test_description_restoration_failure_modes_are_detected():
    original = obj("DESC A", np.nan, 21494, "DESC B")
    same = obj("DESC A", np.nan, 21494, "DESC B")
    as_text = obj("DESC A", np.nan, "21494", "DESC B")
    none_for_nan = obj("DESC A", None, 21494, "DESC B")
    numpy_int = obj("DESC A", np.nan, np.int64(21494), "DESC B")
    assert single(original) == single(same)
    assert single(original) != single(as_text)
    assert single(original) != single(none_for_nan)
    with pytest.raises(FingerprintError, match="unsupported value type"):
        single(numpy_int)


@pytest.mark.parametrize(
    "value",
    [np.int64(5), np.float64(1.5), Decimal("1.5"), datetime.date(2020, 1, 1), [1, 2]],
    ids=["numpy_int", "numpy_float", "decimal", "date", "list"],
)
def test_unsupported_value_types_raise(value):
    with pytest.raises(FingerprintError, match="unsupported value type"):
        single(pd.Series([value], dtype=object))


def test_numpy_float_nan_in_an_object_column_is_rejected_by_exact_type_rule():
    with pytest.raises(FingerprintError, match="unsupported value type float64"):
        single(pd.Series(["a", np.float64("nan")], dtype=object))


def test_unsupported_value_error_reports_the_first_row_position():
    series = obj("a", "b", Decimal("1"), Decimal("2"))
    with pytest.raises(FingerprintError, match="row position 2"):
        fp(pd.DataFrame({"c": series}), chunk_rows=2)


# ------------------- every SUPPORTED dtype, and the unsupported ones -----------
def numbers(dtype):
    return lambda values: pd.Series(values, dtype=dtype)


def stamps(unit):
    return lambda values: pd.Series(pd.to_datetime(values)).astype(f"datetime64[{unit}]")


INTS = [1, 2, 3]
INTS_CHANGED = [1, 2, 4]
FLOATS = [1.5, np.nan, 3.25]
FLOATS_CHANGED = [1.5, np.nan, 3.5]
TIMES = ["2020-01-01 10:00:00", None, "2021-06-30 23:59:59"]
TIMES_CHANGED = ["2020-01-01 10:00:00", None, "2021-06-30 23:59:58"]
WORDS = ["a", None, "b"]
WORDS_CHANGED = ["a", None, "c"]

SUPPORTED_CASES = (
    [
        pytest.param(numbers(dtype), INTS, INTS_CHANGED, dtype, id=dtype)
        for dtype in (
            "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        )
    ]
    + [
        pytest.param(numbers(dtype), FLOATS, FLOATS_CHANGED, dtype, id=dtype)
        for dtype in ("float16", "float32", "float64")
    ]
    + [
        pytest.param(
            numbers("bool"), [True, False, True], [True, False, False], "bool", id="bool"
        )
    ]
    + [
        pytest.param(
            stamps(unit), TIMES, TIMES_CHANGED, f"datetime64[{unit}]", id=f"datetime64_{unit}"
        )
        for unit in ("s", "ms", "us", "ns")
    ]
    + [
        pytest.param(numbers(object), WORDS, WORDS_CHANGED, "values", id="object"),
        pytest.param(lambda v: pd.Series(v), WORDS, WORDS_CHANGED, "values", id="inferred_text"),
        pytest.param(numbers("string"), WORDS, WORDS_CHANGED, "values", id="string_na"),
        pytest.param(
            numbers(pd.StringDtype(na_value=np.nan)), WORDS, WORDS_CHANGED, "values", id="str_nan"
        ),
        pytest.param(
            lambda v: pd.Series(pd.Categorical(v)), WORDS, WORDS_CHANGED, "values", id="categorical"
        ),
    ]
)


@pytest.mark.parametrize("build,base,changed,kind", SUPPORTED_CASES)
def test_every_supported_dtype_is_fingerprinted(build, base, changed, kind):
    first = fp(pd.DataFrame({"c": build(base)}))
    assert first.columns == (("c", kind),)
    assert fp(pd.DataFrame({"c": build(base)})) == first
    assert fp(pd.DataFrame({"c": build(changed)})).digest != first.digest


UNSUPPORTED_COLUMNS = [
    pytest.param(pd.Series(pd.to_timedelta([1, 2], unit="s")), id="timedelta"),
    pytest.param(pd.Series(pd.to_datetime(["2020-01-01"]).tz_localize("UTC")), id="timezone_aware"),
    pytest.param(pd.Series([1, None], dtype="Int64"), id="nullable_integer"),
    pytest.param(pd.Series([1.5, None], dtype="Float64"), id="nullable_float"),
    pytest.param(pd.Series([True, None], dtype="boolean"), id="nullable_boolean"),
    pytest.param(pd.Series(np.array([1 + 2j])), id="complex"),
    pytest.param(pd.Series(["a", None], dtype=pd.ArrowDtype(pa.string())), id="arrow_string"),
    pytest.param(pd.Series(pd.period_range("2020-01", periods=2, freq="M")), id="period"),
]


@pytest.mark.parametrize("series", UNSUPPORTED_COLUMNS)
def test_unsupported_dtypes_raise(series):
    with pytest.raises(FingerprintError, match="unsupported dtype"):
        single(series)


def test_non_string_column_names_raise():
    with pytest.raises(FingerprintError, match="column names must be strings"):
        fp(pd.DataFrame([[1, 2]], columns=[0, 1]))


def test_duplicate_column_names_raise():
    with pytest.raises(FingerprintError, match="duplicate column names"):
        fp(pd.DataFrame([[1, 2]], columns=["a", "a"]))


def test_non_dataframe_input_raises():
    with pytest.raises(FingerprintError, match="DataFrame"):
        fingerprint_dataframe(pd.Series([1]))


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_chunk_rows_raise(bad):
    with pytest.raises(ValueError):
        fp(pd.DataFrame({"a": [1]}), chunk_rows=bad)


# ------------------------- read-only and strided inputs -----------------------
def test_read_only_input_arrays_are_supported():
    numbers_rw = np.arange(3, dtype="int64")
    floats_rw = np.array([1.5, np.nan, 2.5])
    stamps_rw = np.array(["2020-01-01", "NaT", "2021-01-01"], dtype="datetime64[us]")
    read_only = []
    for array in (numbers_rw, floats_rw, stamps_rw):
        copy = array.copy()
        copy.flags.writeable = False
        read_only.append(copy)
    frame_ro = pd.DataFrame({"i": read_only[0], "f": read_only[1], "d": read_only[2]})
    frame_rw = pd.DataFrame({"i": numbers_rw, "f": floats_rw, "d": stamps_rw})
    assert fp(frame_ro) == fp(frame_rw)
    assert not read_only[0].flags.writeable  # the fingerprint never made it writable


def test_columns_backed_by_a_2d_array_match_contiguous_columns():
    matrix = np.arange(12, dtype="int64").reshape(6, 2)
    strided = pd.DataFrame(matrix, columns=["a", "b"])
    contiguous = pd.DataFrame({"a": matrix[:, 0].copy(), "b": matrix[:, 1].copy()})
    assert fp(strided) == fp(contiguous)

    times = np.arange(8, dtype="int64").reshape(4, 2).view("datetime64[us]")
    strided_times = pd.DataFrame(times, columns=["a", "b"])
    contiguous_times = pd.DataFrame(
        {"a": times[:, 0].copy(), "b": times[:, 1].copy()}
    )
    assert fp(strided_times) == fp(contiguous_times)


# ----------------------------- frame-level properties -------------------------
def mixed_frame(n: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": np.arange(n, dtype="int64"),
            "f": np.where(np.arange(n) % 5 == 0, np.nan, np.arange(n) / 4),
            "b": np.arange(n) % 2 == 0,
            "d": pd.date_range("2020-01-01", periods=n, freq="h").astype("datetime64[us]"),
            "s": [f"item {k}" for k in range(n)],
            "o": [k if k % 7 == 0 else (f"x{k}" if k % 3 else np.nan) for k in range(n)],
            "c": pd.Categorical([("u", "v", "w")[k % 3] if k % 4 else None for k in range(n)]),
        }
    )


def test_column_order_matters():
    frame = mixed_frame()
    assert fp(frame[["i", "s"]]).digest != fp(frame[["s", "i"]]).digest


def test_column_name_matters():
    frame = mixed_frame()
    assert fp(frame[["i"]]).digest != fp(frame[["i"]].rename(columns={"i": "j"})).digest


def test_row_order_matters():
    frame = mixed_frame()
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    assert fp(frame).digest != fp(reversed_frame).digest


def test_row_count_matters():
    frame = mixed_frame()
    assert fp(frame).digest != fp(frame.iloc[:-1]).digest
    assert fp(frame).rows == 25
    assert fp(frame.iloc[:-1]).rows == 24


def test_index_labels_are_not_part_of_the_fingerprint():
    frame = mixed_frame()
    shifted = frame.copy()
    shifted.index = range(100, 100 + len(frame))
    assert fp(frame).digest == fp(shifted).digest


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 7, 24, 25, 26, 100, 100_000])
def test_chunk_size_does_not_change_the_fingerprint(chunk_rows):
    frame = mixed_frame()
    assert fp(frame, chunk_rows=chunk_rows) == fp(frame, chunk_rows=100_000)


def every_dtype_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)

    def times(unit):
        values = pd.Series(pd.to_datetime(rng.integers(0, 10**9, n), unit="s"))
        return values.astype(f"datetime64[{unit}]").where(rng.random(n) > 0.1)

    return pd.DataFrame(
        {
            "i8": rng.integers(-100, 100, n).astype("int8"),
            "u64": rng.integers(0, 2**62, n).astype("uint64"),
            "i64": rng.integers(-(2**40), 2**40, n),
            "f16": rng.normal(size=n).astype("float16"),
            "f32": rng.normal(size=n).astype("float32"),
            "f64": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
            "b": rng.random(n) < 0.5,
            "d_s": times("s"),
            "d_ms": times("ms"),
            "d_us": times("us"),
            "d_ns": times("ns"),
            "str": pd.Series([pick.choice(["x", "yy", None, "", "\u00e9"]) for _ in range(n)]),
            "string": pd.Series([pick.choice(["x", None, "zz"]) for _ in range(n)], dtype="string"),
            "obj": pd.Series(
                [pick.choice([1, "1", 2.5, np.nan, None, True, b"b", pd.NA]) for _ in range(n)],
                dtype=object,
            ),
            "cat": pd.Categorical([pick.choice(["u", "v", None]) for _ in range(n)]),
        }
    )


@pytest.mark.parametrize("rows", [0, 1, 9, 600])
@pytest.mark.parametrize("seed", [7, 11])
def test_random_frames_with_every_dtype_ignore_the_chunk_size(rows, seed):
    frame = every_dtype_frame(rows, seed)
    reference = fp(frame, chunk_rows=100_000)
    for chunk_rows in (1, 2, 3, 5, 64, 599, 600, 601):
        assert fp(frame, chunk_rows=chunk_rows) == reference


def test_input_is_not_modified():
    frame = mixed_frame()
    before = frame.copy()
    fp(frame)
    pd.testing.assert_frame_equal(before, frame)


def test_empty_frames():
    empty = pd.DataFrame(
        {"x": pd.Series([], dtype="int64"), "y": pd.Series([], dtype=object)}
    )
    result = fp(empty)
    assert result.rows == 0
    assert len(result.digest) == 64
    assert fp(empty) == result
    other = pd.DataFrame(
        {"x": pd.Series([], dtype="float64"), "y": pd.Series([], dtype=object)}
    )
    assert fp(other).digest != result.digest


def test_zero_column_frames():
    three = fp(pd.DataFrame(index=range(3)))
    assert three.rows == 3
    assert three.columns == ()
    assert three.digest != fp(pd.DataFrame(index=range(4))).digest


def test_column_seconds_are_recorded():
    seconds: dict[str, float] = {}
    fp(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), column_seconds=seconds)
    assert list(seconds) == ["a", "b"]
    assert all(isinstance(value, float) and value >= 0 for value in seconds.values())


def test_differing_columns_reports_changed_column():
    first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    second = pd.DataFrame({"a": [1, 2], "b": ["x", "z"]})
    assert differing_columns(fp(first), fp(second)) == ("b",)


def test_differing_columns_reports_added_and_removed_columns():
    first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    wider = first.assign(c=[3, 4])
    assert differing_columns(fp(first), fp(wider)) == ("c",)
    assert differing_columns(fp(wider), fp(first)) == ("c",)
    changed_and_added = wider.assign(b=["x", "z"])
    assert differing_columns(fp(first), fp(changed_and_added)) == ("b", "c")


def test_differing_columns_is_empty_for_identical_fingerprints():
    frame = mixed_frame()
    assert differing_columns(fp(frame), fp(frame.copy())) == ()


def test_categorical_missing_is_nan_not_none():
    categorical = pd.Series(pd.Categorical(["a", None]))
    assert single(categorical) == single(obj("a", np.nan))
    assert single(categorical) != single(obj("a", None))


# ---------------- integrity and privacy: no per-column digest is needed ---------
def frame_with_identifier_column(n: int = 25) -> pd.DataFrame:
    frame = mixed_frame(n)
    frame["Customer ID"] = np.where(np.arange(n) % 4 == 1, np.nan, 12000.0 + np.arange(n))
    return frame


def mutate_last_cell(series: pd.Series) -> pd.Series:
    changed = series.copy()
    dtype = changed.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        changed = changed.cat.add_categories(["__changed__"])
        changed.iloc[-1] = "__changed__"
    elif dtype == np.dtype(bool):
        changed.iloc[-1] = not changed.iloc[-1]
    elif isinstance(dtype, np.dtype) and dtype.kind in "iuf":
        changed.iloc[-1] = changed.iloc[-1] + 1
    elif isinstance(dtype, np.dtype) and dtype.kind == "M":
        changed.iloc[-1] = changed.iloc[-1] + pd.Timedelta(seconds=1)
    else:
        changed.iloc[-1] = "__changed__"
    return changed


@pytest.mark.parametrize("column", list(frame_with_identifier_column().columns))
def test_changing_one_cell_in_any_column_alters_the_outer_digest(column):
    frame = frame_with_identifier_column()
    changed = frame.copy()
    changed[column] = mutate_last_cell(frame[column])
    assert not changed[column].equals(frame[column])
    assert fp(changed).digest != fp(frame).digest
    assert differing_columns(fp(frame), fp(changed)) == (column,)


def test_manifest_dict_holds_only_the_outer_digest_and_column_kinds():
    result = fp(frame_with_identifier_column())
    manifest = result.to_manifest_dict()
    assert set(manifest) == {"fingerprint_version", "digest", "rows", "columns"}
    assert manifest["fingerprint_version"] == FINGERPRINT_VERSION
    assert manifest["digest"] == result.digest
    assert manifest["rows"] == 25
    assert manifest["columns"] == [[name, kind] for name, kind in result.columns]
    assert ["Customer ID", "float64"] in manifest["columns"]
    text = canonical_json_bytes(manifest).decode("ascii")
    for _, column_digest in result.column_digests:
        assert column_digest not in text


def test_repr_and_str_do_not_show_per_column_digests():
    result = fp(frame_with_identifier_column())
    for text in (repr(result), str(result)):
        assert "column_digests" not in text
        for _, column_digest in result.column_digests:
            assert column_digest not in text
        assert result.digest in text  # the outer digest is the one shown


def test_differing_columns_returns_names_never_digests():
    first = frame_with_identifier_column()
    changed = first.copy()
    changed["Customer ID"] = mutate_last_cell(first["Customer ID"])
    names = differing_columns(fp(first), fp(changed))
    assert names == ("Customer ID",)
    assert all(isinstance(name, str) and len(name) != 64 for name in names)