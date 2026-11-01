import sqlite3
from pathlib import Path

import numpy as np
from hpk.clean import clean
from hpk.modeling import FEATURES, MODEL_NAMES, TARGET, add_features, build_model
from hpk.synth import make_synthetic
from hpk.valuation import score_listings


def _data():
    df, _ = clean(make_synthetic(1200, seed=3))
    return add_features(df)


def test_all_models_fit_and_predict():
    df = _data()
    for name in MODEL_NAMES:
        m = build_model(name).fit(df[FEATURES], df[TARGET])
        assert np.isfinite(m.predict(df[FEATURES])).all()


def test_unknown_area_does_not_crash():
    df = _data()
    m = build_model(MODEL_NAMES[0]).fit(df[FEATURES], df[TARGET])
    row = df.head(1).copy(); row["area"] = "Nowhere | Nowhere Town"
    assert np.isfinite(m.predict(row[FEATURES])).all()


def test_valuation_flags_a_planted_bargain_and_suspects():
    df, _ = clean(make_synthetic(1500, seed=4))
    df.loc[0, "price_pkr"] *= 0.6; df.loc[0, "log_price"] = np.log(df.loc[0, "price_pkr"])
    df.loc[1, "price_pkr"] *= 0.1; df.loc[1, "log_price"] = np.log(df.loc[1, "price_pkr"])
    out = score_listings(df, MODEL_NAMES[0], under=0.8, suspect_low=0.45, min_area_listings=30)
    assert out.loc[0, "verdict"] == "Undervalued"
    assert out.loc[1, "verdict"] == "Check data"
    assert (out["confidence"] == "ok").mean() > 0.9


def test_sql_analysis_runs():
    df, _ = clean(make_synthetic(1500, seed=5))
    con = sqlite3.connect(":memory:"); df.to_sql("listings", con, index=False)
    sql = Path(__file__).parents[1].joinpath("sql/analysis.sql").read_text()
    stmts = [s for s in sql.split(";") if s.strip() and not all(l.strip().startswith("--") or not l.strip() for l in s.splitlines())]
    assert len(stmts) == 4
    for s in stmts:
        assert len(con.execute(s).fetchall()) > 0
