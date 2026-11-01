"""Train and compare models, run error analysis, save the best one.

    python -m hpk.train --data data/processed/listings_clean.csv --trials 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split

from .modeling import FEATURES, MODEL_NAMES, TARGET, XGB_NAME, HAS_XGB, add_features, build_model, regression_metrics


def tune_xgb(X, y, trials: int, seed: int) -> dict:
    """Optuna search over XGBoost params, scored by 3-fold CV RMSE on log price (train only)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 200, 1200, step=100),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 max_depth=t.suggest_int("max_depth", 3, 10),
                 subsample=t.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                 min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-2, 10, log=True))
        m = build_model(XGB_NAME, p, seed)
        return -cross_val_score(m, X, y, cv=3, scoring="neg_root_mean_squared_error").mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return study.best_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/clean_2024.csv")
    ap.add_argument("--trials", type=int, default=0, help="Optuna trials for XGBoost (0 = skip tuning)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    out_m, out_r = Path(a.models_dir), Path(a.reports_dir)
    out_m.mkdir(exist_ok=True, parents=True); out_r.mkdir(exist_ok=True, parents=True)

    df = add_features(pd.read_csv(a.data))
    train, test = train_test_split(df, test_size=0.2, random_state=a.seed, stratify=df["city"])
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    best_params = {}
    if a.trials > 0 and HAS_XGB:
        try:
            best_params = tune_xgb(Xtr, ytr, a.trials, a.seed)
            print("Optuna best params:", best_params)
        except ImportError:
            print("optuna not installed, using default XGBoost params")

    rows, fitted, cv_scores = [], {}, {}
    for name in MODEL_NAMES:
        model = build_model(name, best_params if name == XGB_NAME else None, a.seed)
        cv_scores[name] = float(-cross_val_score(model, Xtr, ytr, cv=3, scoring="neg_root_mean_squared_error").mean())
        model.fit(Xtr, ytr)
        fitted[name] = model
        pred = np.exp(model.predict(Xte))
        rows.append({"model": name, "cv_rmse_log": cv_scores[name], **regression_metrics(np.exp(yte), pred)})
    metrics = pd.DataFrame(rows).sort_values("RMSE_pkr")
    metrics.to_csv(out_r / "metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))

    # model choice uses CV on the training set only, never the test set
    best_name = min(cv_scores, key=cv_scores.get)
    best = fitted[best_name]
    lin = metrics.set_index("model").loc["Linear Regression"]
    cur = metrics.set_index("model").loc[best_name]
    print(f"\nSelected: {best_name}. RMSE improvement vs linear: {(1 - cur.RMSE_pkr / lin.RMSE_pkr) * 100:.1f}%")

    # error analysis on the held-out set
    t = test.copy()
    t["pred"] = np.exp(best.predict(Xte)); t["actual"] = np.exp(yte)
    t["abs_err"] = (t.pred - t.actual).abs(); t["ape"] = t.abs_err / t.actual * 100
    by_city = t.groupby("city").apply(lambda g: pd.Series({"n": len(g), **regression_metrics(g.actual, g.pred)}), include_groups=False)
    t["price_bucket"] = pd.qcut(t["actual"], 5, labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 priciest"])
    # R2 is meaningless inside a narrow price band, so report only error sizes here
    by_price = t.groupby("price_bucket", observed=True).apply(
        lambda g: pd.Series({"n": len(g), **{k: v for k, v in regression_metrics(g.actual, g.pred).items() if k != "R2"}}),
        include_groups=False)
    by_city.to_csv(out_r / "error_by_city.csv"); by_price.to_csv(out_r / "error_by_price_bucket.csv")
    print("\nError by city:\n", by_city.round(2).to_string())
    print("\nError by price bucket:\n", by_price.round(2).to_string())

    # prediction interval from out-of-fold residuals on the TRAIN set (log scale)
    oof = cross_val_predict(build_model(best_name, best_params if best_name == XGB_NAME else None, a.seed),
                            Xtr, ytr, cv=KFold(5, shuffle=True, random_state=a.seed))
    resid = ytr.values - oof
    lo, hi = np.quantile(resid, [0.10, 0.90])

    joblib.dump(best, out_m / "model.joblib")
    areas = df.groupby("city")["location"].agg(lambda s: sorted(s.unique())).to_dict()
    stats = df.groupby(["city", "location"]).agg(
        listings=("price_pkr", "size"), median_price_pkr=("price_pkr", "median"),
        median_price_per_sqft=("price_per_sqft", "median"),
        **({"lat": ("latitude", "mean"), "lon": ("longitude", "mean")} if "latitude" in df.columns else {})).reset_index()
    stats[stats.listings >= 5].to_csv(out_m / "area_stats.csv", index=False)  # aggregates only, safe to publish
    meta = {"model": best_name, "features": FEATURES, "interval_log": [float(lo), float(hi)],
            "areas_by_city": areas, "n_train": len(train), "n_test": len(test),
            "metrics": metrics.set_index("model").to_dict("index"), "xgb_params": best_params}
    (out_m / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nsaved", out_m / "model.joblib")


if __name__ == "__main__":
    main()
