import numpy as np
import pandas as pd
import hpk.explain as ex
from hpk.clean import clean
from hpk.modeling import FEATURES, MODEL_NAMES, TARGET, add_features, build_model
from hpk.synth import make_synthetic


class FakeExplainer:
    """Stands in for shap.TreeExplainer (shap is not installed in every environment)."""
    expected_value = np.array([14.5])

    def __init__(self, n_features):
        self.n = n_features

    def shap_values(self, X):
        return np.ones((len(X), self.n))  # +1 for every transformed column


def test_shap_with_base_groups_columns_back_to_original_features(monkeypatch):
    df, _ = clean(make_synthetic(900, seed=11), min_city_rows=0)
    df = add_features(df)
    pipe = build_model(MODEL_NAMES[0]).fit(df[FEATURES], df[TARGET])
    n = len(pipe.named_steps["prep"].get_feature_names_out())
    monkeypatch.setattr(ex, "_explainer", lambda p: FakeExplainer(n))
    ex._EXPLAINERS.clear()
    contrib, base = ex.shap_with_base(pipe, df[FEATURES].head(1))
    assert base == 14.5
    assert {"city", "area", "area_sub", "property_type", "log_area", "bedrooms"} <= set(contrib.columns)
    # area_sub columns must not be swallowed by "area" (longest-prefix matching)
    onehot = pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out()
    n_sub = sum(1 for c in onehot if c.startswith("area_sub_"))
    assert contrib.loc[0, "area_sub"] == n_sub and contrib.loc[0, "log_area"] == 1.0
    assert contrib.loc[0].sum() == n   # nothing lost or double counted
    assert ex.shap_by_feature(pipe, df[FEATURES].head(1)).shape == contrib.shape


def test_returns_none_when_shap_unavailable():
    df, _ = clean(make_synthetic(600, seed=12), min_city_rows=0); df = add_features(df)
    pipe = build_model(MODEL_NAMES[0]).fit(df[FEATURES], df[TARGET])
    ex._EXPLAINERS.clear()
    assert ex.shap_with_base(pipe, df[FEATURES].head(1)) is None   # no shap here, so it degrades quietly
