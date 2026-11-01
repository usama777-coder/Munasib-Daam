"""SHAP explanations grouped back to the original features (city, area, size...).

Works for tree models (XGBoost, Random Forest). Returns None when SHAP is not installed or the model type is unsupported.
Contributions are on the log-price scale and add up to: log(estimate) - base, where base = the average listing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .modeling import CAT_COLS, NUM_COLS

_EXPLAINERS: dict = {}


def _group(feature_name: str) -> str:
    name = feature_name.split("__", 1)[-1]
    for c in sorted(CAT_COLS + NUM_COLS, key=len, reverse=True):  # longest first: area_sub before area
        if name == c or name.startswith(c + "_"):
            return c
    return name


def _explainer(pipe):
    import shap
    return shap.TreeExplainer(pipe.named_steps["model"])


def shap_with_base(pipe, X: pd.DataFrame):
    """Return (contributions per original feature, base log price) or None."""
    try:
        key = id(pipe)
        if key not in _EXPLAINERS:
            _EXPLAINERS[key] = _explainer(pipe)
        ex = _EXPLAINERS[key]
        Xt = pipe.named_steps["prep"].transform(X)
        sv = ex.shap_values(Xt)
        base = float(np.ravel(ex.expected_value)[0])
    except Exception:
        return None
    names = pipe.named_steps["prep"].get_feature_names_out()
    df = pd.DataFrame(sv, columns=[_group(n) for n in names]).T.groupby(level=0).sum().T
    return df, base


def shap_by_feature(pipe, X: pd.DataFrame) -> pd.DataFrame | None:
    res = shap_with_base(pipe, X)
    return None if res is None else res[0]


def as_percent(contrib: pd.Series) -> pd.Series:
    """log-price contribution -> approximate % effect on price."""
    return (np.exp(contrib) - 1) * 100


def global_importance(pipe, X_sample: pd.DataFrame) -> pd.Series | None:
    sv = shap_by_feature(pipe, X_sample)
    return None if sv is None else sv.abs().mean().sort_values(ascending=False)
