"""Undervalued/overpriced finder and price trends.

Predictions here are OUT-OF-FOLD: each listing is scored by a model that never saw it,
otherwise the model would just memorise listed prices and flag nothing.

    python -m hpk.valuation --data data/processed/listings_clean.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from .modeling import FEATURES, TARGET, add_features, build_model


def score_listings(df: pd.DataFrame, model_name: str, params: dict | None = None, seed: int = 42,
                   under: float = 0.80, over: float = 1.25, suspect_low: float = 0.0,
                   suspect_high: float = 1e9, min_area_listings: int = 0) -> pd.DataFrame:
    """Verdicts: Check data (extreme gap, probably an error) > Undervalued / Overpriced > Fair.
    area_listings = how many listings share this city+location (few = low confidence)."""
    df = add_features(df)
    oof = cross_val_predict(build_model(model_name, params, seed), df[FEATURES], df[TARGET],
                            cv=KFold(5, shuffle=True, random_state=seed))
    out = df.copy()
    out["fair_price_pkr"] = np.exp(oof)
    out["price_ratio"] = out["price_pkr"] / out["fair_price_pkr"]
    out["gap_pkr"] = out["price_pkr"] - out["fair_price_pkr"]
    out["area_listings"] = out.groupby(["city", "location"])["price_pkr"].transform("size")
    out["verdict"] = np.select([(out.price_ratio < suspect_low) | (out.price_ratio > suspect_high),
                                out.price_ratio < under, out.price_ratio > over],
                               ["Check data", "Undervalued", "Overpriced"], "Fair")
    out["confidence"] = np.where(out.area_listings >= min_area_listings, "ok", "low (few listings in this area)")
    return out


def price_trends(df: pd.DataFrame, min_listings: int = 5) -> pd.DataFrame:
    if "listing_month" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["city", "location", "listing_month"]).agg(
        listings=("price_pkr", "size"), median_price_per_sqft=("price_per_sqft", "median")).reset_index()
    return g[g.listings >= min_listings].sort_values(["city", "location", "listing_month"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/clean_2024.csv")
    ap.add_argument("--meta", default="models/meta.json")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--under", type=float, default=0.70, help="flag if listed price < under x fair price")
    ap.add_argument("--over", type=float, default=1.40, help="flag if listed price > over x fair price")
    ap.add_argument("--suspect-low", type=float, default=0.45, help="below this ratio = 'Check data' (probably an error)")
    ap.add_argument("--suspect-high", type=float, default=2.2, help="above this ratio = 'Check data'")
    ap.add_argument("--min-area-listings", type=int, default=30, help="areas with fewer listings are low confidence")
    a = ap.parse_args()
    meta = json.loads(Path(a.meta).read_text())
    df = pd.read_csv(a.data)
    scored = score_listings(df, meta["model"], meta.get("xgb_params") or None, under=a.under, over=a.over,
                            suspect_low=a.suspect_low, suspect_high=a.suspect_high,
                            min_area_listings=a.min_area_listings)
    out = Path(a.reports_dir); out.mkdir(exist_ok=True, parents=True)
    cols = [c for c in ["city", "location", "property_type", "area_sqft", "bedrooms", "price_pkr",
                        "fair_price_pkr", "price_ratio", "gap_pkr", "verdict", "confidence", "area_listings", "url"] if c in scored.columns]
    scored[cols].to_csv(out / "scored_listings.csv", index=False)
    top = scored[(scored.verdict == "Undervalued") & (scored.confidence == "ok")].sort_values("price_ratio").groupby("city").head(a.top)
    top[cols].to_csv(out / "undervalued_listings.csv", index=False)
    tr = price_trends(scored)
    if len(tr):
        tr.to_csv(out / "price_trends.csv", index=False)
    print(scored.verdict.value_counts().to_string())
    print("\nVerdict by city:\n", pd.crosstab(scored.city, scored.verdict, normalize="index").round(3).to_string())


if __name__ == "__main__":
    main()
