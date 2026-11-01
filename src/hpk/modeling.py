"""Shared model definitions. Used by training, valuation and the app."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .textfeats import FLAG_PATTERNS, FLAGS, slugs_for, subarea

CAT_COLS = ["city", "area", "area_sub", "property_type"]
NUM_COLS = ["log_area", "bedrooms", "bathrooms"] + FLAGS
FEATURES = CAT_COLS + NUM_COLS
TARGET = "log_price"

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

XGB_NAME = "XGBoost" if HAS_XGB else "HistGradientBoosting (xgboost not installed)"
DEFAULT_XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """area = city + location ('Bahria Town' exists in several cities); area_sub adds phase/block/sector.
    subarea and keyword flags are parsed from the URL slug unless the caller already supplies them (the app does)."""
    out = df.copy()
    out["area"] = out["city"].astype(str) + " | " + out["location"].astype(str)
    slugs = slugs_for(out)
    if "subarea" not in out.columns:
        out["subarea"] = slugs.map(subarea)
    for f, pat in FLAG_PATTERNS.items():
        if f not in out.columns:
            out[f] = slugs.map(lambda t, p=pat: int(re.search(p, t) is not None))
    out["area_sub"] = out["area"] + " | " + out["subarea"].astype(str)
    if "log_area" not in out.columns and "area_sqft" in out.columns:
        out["log_area"] = np.log(out["area_sqft"])
    return out


def _preprocessor(min_freq: int = 15) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq,
                              sparse_output=False), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])


def build_model(name: str, xgb_params: dict | None = None, seed: int = 42) -> Pipeline:
    if name == "Linear Regression":
        est = LinearRegression()
    elif name == "Random Forest":
        est = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    elif name == XGB_NAME:
        if HAS_XGB:
            est = XGBRegressor(**{**DEFAULT_XGB, **(xgb_params or {})}, random_state=seed, n_jobs=-1)
        else:
            est = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("prep", _preprocessor()), ("model", est)])


MODEL_NAMES = ["Linear Regression", "Random Forest", XGB_NAME]


def regression_metrics(y_true_pkr, y_pred_pkr) -> dict:
    err = np.asarray(y_pred_pkr) - np.asarray(y_true_pkr)
    ss_res, ss_tot = float((err ** 2).sum()), float(((y_true_pkr - np.mean(y_true_pkr)) ** 2).sum())
    return {
        "RMSE_pkr": float(np.sqrt(np.mean(err ** 2))),
        "MAE_pkr": float(np.mean(np.abs(err))),
        "R2": 1 - ss_res / ss_tot,
        "MedianAPE_pct": float(np.median(np.abs(err) / y_true_pkr) * 100),
    }
