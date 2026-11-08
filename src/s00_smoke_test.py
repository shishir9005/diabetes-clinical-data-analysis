"""Environment smoke test (Day 1, step 2).

Checks that every approved package imports from the new environment and that
a few tiny, non-modelling operations work. It does NOT fit any model.

Run with:  <env python> -I -B src/s00_smoke_test.py
Writes:    reports/00_smoke_test.json
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "reports" / "00_smoke_test.json"

results: dict = {
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "sys_prefix": sys.prefix,
    "isolated_mode_flag": bool(sys.flags.isolated),
    "checks": {},
}


def check(name, fn):
    try:
        detail = fn()
        results["checks"][name] = {"ok": True, "detail": detail}
    except Exception as exc:  # report every failure instead of stopping at the first
        results["checks"][name] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def _numpy():
    import numpy as np
    assert int(np.arange(5).sum()) == 10
    return np.__version__


def _pandas():
    import pandas as pd
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    assert df.groupby("g")["v"].sum().to_dict() == {"a": 3, "b": 3}
    return pd.__version__


def _sklearn():
    # Import and split toy data only: splitting is not model training.
    import numpy as np
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.zeros((12, 1))
    y = np.array([0, 1] * 6)
    groups = np.repeat(np.arange(6), 2)
    n_folds = sum(1 for _ in StratifiedGroupKFold(n_splits=3).split(X, y, groups))
    assert n_folds == 3
    return sklearn.__version__


def _scipy():
    import scipy
    return scipy.__version__


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")  # rendered in memory, nothing written to disk
    plt.close(fig)
    assert buf.tell() > 0
    return matplotlib.__version__


def _seaborn():
    import seaborn as sns
    return sns.__version__


def _joblib():
    import joblib
    buf = io.BytesIO()
    joblib.dump({"x": 1}, buf)
    buf.seek(0)
    assert joblib.load(buf) == {"x": 1}
    return joblib.__version__


def _streamlit():
    import streamlit
    return streamlit.__version__


def _jupyterlab():
    import jupyterlab
    return jupyterlab.__version__


def _sqlite():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (a INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    total = con.execute("SELECT SUM(a) FROM t").fetchone()[0]
    con.close()
    assert total == 6
    return sqlite3.sqlite_version


for name, fn in [
    ("numpy", _numpy), ("pandas", _pandas), ("scikit-learn", _sklearn),
    ("scipy", _scipy), ("matplotlib", _matplotlib), ("seaborn", _seaborn),
    ("joblib", _joblib), ("streamlit", _streamlit), ("jupyterlab", _jupyterlab),
    ("sqlite3", _sqlite),
]:
    check(name, fn)

results["all_ok"] = all(c["ok"] for c in results["checks"].values())
results["prefix_is_new_env"] = sys.prefix.endswith("/envs/diabetes-ml")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
sys.exit(0 if results["all_ok"] and results["prefix_is_new_env"] else 1)
