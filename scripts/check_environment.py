"""Environment check for the E-Commerce Customer Behavior Analysis System.

Local-only: makes no network calls, writes no files, and reads no environment
variables. Run from the project root:

    python scripts/check_environment.py

Exit code 0 means every check passed; 1 means at least one failed.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import io
import sys
from dataclasses import dataclass
from typing import Callable

MIN_PYTHON: tuple[int, int] = (3, 11)

# (import name, distribution name on PyPI)
PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("mlxtend", "mlxtend"),
    ("plotly", "plotly"),
    ("streamlit", "streamlit"),
    ("yaml", "PyYAML"),
    ("joblib", "joblib"),
    ("openpyxl", "openpyxl"),
    ("pytest", "pytest"),
]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run(name: str, func: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name, True, func())
    except Exception as exc:  # noqa: BLE001 - report any failure, never crash
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------
def _check_python() -> str:
    if sys.version_info[:2] < MIN_PYTHON:
        raise RuntimeError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
            f"found {sys.version.split()[0]}"
        )
    return f"Python {sys.version.split()[0]}"


def _check_venv() -> str:
    in_venv = sys.prefix != sys.base_prefix
    return f"virtual environment active: {'yes' if in_venv else 'NO (recommended: yes)'}"


def _check_import(import_name: str, dist_name: str) -> str:
    importlib.import_module(import_name)
    return f"version {md.version(dist_name)}"


# --------------------------------------------------------------------------
# Functional smoke checks (tiny, deterministic, hand-verifiable)
# --------------------------------------------------------------------------
def _smoke_pandas() -> str:
    import pandas as pd

    df = pd.DataFrame(
        {
            "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-02 09:00:00"],
            "Quantity": [2, 3],
            "Price": [1.5, 2.0],
        }
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")
    df["Revenue"] = df["Quantity"] * df["Price"]
    assert df["Revenue"].sum() == 9.0
    assert str(df["InvoiceDate"].dtype).startswith("datetime64")
    string_dtype = pd.Series(["a"]).dtype
    return f"datetime parsing + arithmetic OK (default string dtype: {string_dtype})"


def _smoke_kmeans() -> str:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = np.array(
        [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float
    )
    first = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(x)
    second = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(x)
    assert (first == second).all(), "KMeans not reproducible with fixed seed"
    assert len(set(first.tolist())) == 2
    score = float(silhouette_score(x, first))
    assert score > 0.5, f"unexpected silhouette {score}"
    return f"reproducible with random_state=42, silhouette={score:.3f}"


def _smoke_classifiers() -> str:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    x = np.array([[0.0], [0.1], [0.2], [0.3], [0.8], [0.9], [1.0], [1.1]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=20, random_state=42),
    }
    for name, model in models.items():
        model.fit(x, y)
        proba = model.predict_proba(x)[:, 1]
        assert roc_auc_score(y, proba) > 0.9, f"{name} ROC-AUC too low"
        assert average_precision_score(y, proba) > 0.9, f"{name} PR-AUC too low"
    return "LogisticRegression + RandomForest fit, ROC-AUC and PR-AUC computed"


def _smoke_mlxtend() -> str:
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

    # 4 baskets. Supports: A=0.75, B=0.75, C=0.25, {A,B}=0.50
    baskets = pd.DataFrame(
        {
            "A": [True, True, True, False],
            "B": [True, True, False, True],
            "C": [False, True, False, False],
        }
    )
    expected = {frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"})}

    freq_apriori = apriori(baskets, min_support=0.4, use_colnames=True)
    freq_fpgrowth = fpgrowth(baskets, min_support=0.4, use_colnames=True)
    assert {frozenset(s) for s in freq_apriori["itemsets"]} == expected
    assert {frozenset(s) for s in freq_fpgrowth["itemsets"]} == expected

    params = inspect.signature(association_rules).parameters
    kwargs = {}
    if "num_itemsets" in params:
        kwargs["num_itemsets"] = len(baskets)
    rules = association_rules(
        freq_apriori, metric="confidence", min_threshold=0.5, **kwargs
    )

    is_a = rules["antecedents"].apply(lambda s: frozenset(s) == frozenset({"A"}))
    is_b = rules["consequents"].apply(lambda s: frozenset(s) == frozenset({"B"}))
    row = rules[is_a & is_b]
    assert len(row) == 1, "rule A -> B not found"
    expected_conf = 0.50 / 0.75
    expected_lift = expected_conf / 0.75
    assert abs(float(row["confidence"].iloc[0]) - expected_conf) < 1e-9
    assert abs(float(row["lift"].iloc[0]) - expected_lift) < 1e-9
    return (
        "apriori/fpgrowth agree; A->B confidence and lift match hand calculation "
        f"(association_rules takes num_itemsets: {'num_itemsets' in params})"
    )


def _smoke_plotly() -> str:
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))
    assert '"bar"' in fig.to_json()
    return "figure built and serialised (no network, no rendering)"


def _smoke_yaml() -> str:
    import yaml

    assert yaml.safe_load("a: 1\nb: [1, 2]") == {"a": 1, "b": [1, 2]}
    return "yaml.safe_load OK"


def _smoke_excel() -> str:
    import pandas as pd

    frame = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    loaded = pd.read_excel(buffer, engine="openpyxl")
    assert loaded["x"].tolist() == [1, 2]
    assert loaded["y"].tolist() == ["a", "b"]
    return "in-memory Excel write/read round trip OK"


def _smoke_joblib() -> str:
    import joblib

    buffer = io.BytesIO()
    joblib.dump({"k": [1, 2, 3]}, buffer)
    buffer.seek(0)
    assert joblib.load(buffer) == {"k": [1, 2, 3]}
    return "in-memory round trip OK (own object only; never load untrusted files)"


SMOKE_CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("smoke:pandas", _smoke_pandas),
    ("smoke:kmeans", _smoke_kmeans),
    ("smoke:classifiers", _smoke_classifiers),
    ("smoke:mlxtend", _smoke_mlxtend),
    ("smoke:plotly", _smoke_plotly),
    ("smoke:yaml", _smoke_yaml),
    ("smoke:excel", _smoke_excel),
    ("smoke:joblib", _smoke_joblib),
]


def run_all_checks() -> list[CheckResult]:
    results = [
        _run("python", _check_python),
        _run("environment", _check_venv),
    ]
    for import_name, dist_name in PACKAGES:
        results.append(
            _run(f"import:{import_name}", lambda i=import_name, d=dist_name: _check_import(i, d))
        )
    for name, func in SMOKE_CHECKS:
        results.append(_run(name, func))
    return results


def main() -> int:
    results = run_all_checks()
    width = max(len(r.name) for r in results)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name.ljust(width)}  {result.detail}")
    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"{len(failed)} of {len(results)} checks FAILED.")
        return 1
    print(f"ALL {len(results)} CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())