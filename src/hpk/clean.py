"""Raw listings -> clean modelling table.

Input columns (rename yours in COLUMN_MAP or pass a mapping):
  city, location, property_type, price_raw, area_raw, bedrooms_raw, bathrooms_raw,
  [latitude, longitude, listing_date, url]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .parsing import parse_area_sqft, parse_count, parse_price
from .textfeats import INSTALLMENT, is_plot_ad, slugs_for

# Normalise spellings; any city not listed here is kept as Title Case (Rawalpindi, Faisalabad, ...)
CITY_ALIASES = {"isb": "Islamabad", "lhr": "Lahore", "khi": "Karachi", "pwr": "Peshawar", "rwp": "Rawalpindi"}
TYPE_ALIASES = {"home": "House", "villa": "House", "apartment": "Flat", "portion": "Upper Portion"}


def _tidy_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


RESIDENTIAL = ["House", "Flat", "Upper Portion", "Lower Portion", "Penthouse"]


def clean(raw: pd.DataFrame, price_q=(0.005, 0.995), ppsf_q=(0.01, 0.99), min_area_sqft=250,
          max_area_sqft=60_000, property_types=None, content_dedupe=True,
          min_city_rows=200) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). The report records how many rows each step removed;
    put it in your README, it shows you know your data."""
    rep = {"rows_raw": len(raw)}
    df = raw.copy()

    city = _tidy_text(df["city"]).str.lower()
    df["city"] = city.map(CITY_ALIASES).fillna(city.str.title()).astype("string")
    df["location"] = _tidy_text(df["location"]).str.title()
    ptype = _tidy_text(df["property_type"]).str.lower()
    df["property_type"] = ptype.map(TYPE_ALIASES).fillna(ptype.str.title()).fillna("Other")

    df["price_pkr"] = df["price_raw"].map(parse_price)
    df["area_sqft"] = df["area_raw"].map(parse_area_sqft)
    df["bedrooms"] = df["bedrooms_raw"].map(parse_count)
    df["bathrooms"] = df["bathrooms_raw"].map(parse_count)

    if property_types:  # e.g. keep houses/flats, drop plots, shops, offices
        rep["property_type_counts_before_filter"] = df["property_type"].value_counts().head(15).to_dict()
        keep_t = df["property_type"].isin(property_types)
        rep["dropped_other_property_types"] = int((~keep_t).sum())
        df = df[keep_t]

    if "url" in df.columns:  # instalment / down-payment ads show a deposit, not the full price
        inst = slugs_for(df).str.contains(INSTALLMENT, regex=True)
        rep["dropped_installment_ads"] = int(inst.sum())
        df = df[~inst]
        plots = slugs_for(df).map(is_plot_ad)  # land ads wrongly typed as House/Flat
        rep["dropped_plot_ads_mislabelled"] = int(plots.sum())
        df = df[~plots]

    # Zameen-style data stores "unknown" as 0 beds / 0 baths; a real 0 only makes sense for a Room
    for col in ["bedrooms", "bathrooms"]:
        zero_as_missing = (df[col] == 0) & (df["property_type"] != "Room")
        rep[f"{col}_zero_set_to_missing"] = int(zero_as_missing.sum())
        df.loc[zero_as_missing, col] = np.nan
    rep["dropped_unknown_city"] = int(df["city"].isna().sum())
    rep["dropped_unparseable_price"] = int(df["price_pkr"].isna().sum())
    rep["dropped_unparseable_area"] = int(df["area_sqft"].isna().sum())
    df = df.dropna(subset=["city", "price_pkr", "area_sqft"])

    before = len(df)
    dedupe_cols = [c for c in ["url"] if c in df.columns] or ["city", "location", "price_pkr", "area_sqft", "bedrooms"]
    df = df.drop_duplicates(subset=dedupe_cols)
    rep["dropped_duplicates"] = before - len(df)

    if content_dedupe:  # same property re-posted as a new ad -> would leak between train and test
        before = len(df)
        df = df.drop_duplicates(subset=["city", "location", "property_type", "price_pkr", "area_sqft",
                                        "bedrooms", "bathrooms"])
        rep["dropped_content_duplicates"] = before - len(df)

    before = len(df)
    df = df[df["area_sqft"].between(min_area_sqft, max_area_sqft)]
    rep["dropped_implausible_area"] = before - len(df)

    if min_city_rows:  # cities with a handful of rows cannot be modelled or evaluated
        counts = df["city"].value_counts()
        small = counts[counts < min_city_rows]
        rep["dropped_small_cities"] = {k: int(v) for k, v in small.items()}
        df = df[~df["city"].isin(small.index)]

    # outliers are judged within each city, so Karachi flats are not judged against Islamabad villas
    df["price_per_sqft"] = df["price_pkr"] / df["area_sqft"]
    g = df.groupby("city")
    lo_p, hi_p = g["price_pkr"].transform(lambda s: s.quantile(price_q[0])), g["price_pkr"].transform(lambda s: s.quantile(price_q[1]))
    lo_s, hi_s = g["price_per_sqft"].transform(lambda s: s.quantile(ppsf_q[0])), g["price_per_sqft"].transform(lambda s: s.quantile(ppsf_q[1]))
    keep = df["price_pkr"].between(lo_p, hi_p) & df["price_per_sqft"].between(lo_s, hi_s)
    rep["dropped_price_outliers"] = int((~keep).sum())
    df = df[keep]

    # impute counts with the median for that city + property type, and keep a flag so the model knows
    for col in ["bedrooms", "bathrooms"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        med = df.groupby(["city", "property_type"])[col].transform("median")
        df[col] = df[col].fillna(med).fillna(df[col].median()).fillna(0).round().astype(int)

    df["log_price"] = np.log(df["price_pkr"])
    df["log_area"] = np.log(df["area_sqft"])
    if "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
        df["listing_month"] = df["listing_date"].dt.to_period("M").astype("string")

    rep["rows_clean"] = len(df)
    rep["rows_per_city"] = df["city"].value_counts().to_dict()
    return df.reset_index(drop=True), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw/listings.csv")
    ap.add_argument("--out", default="data/processed/listings_clean.csv")
    ap.add_argument("--report", default="reports/cleaning_report.json")
    ap.add_argument("--types", default=",".join(RESIDENTIAL),
                    help="comma-separated property types to keep, or 'all'")
    ap.add_argument("--min-city-rows", type=int, default=200)
    ap.add_argument("--keep-content-duplicates", action="store_true")
    a = ap.parse_args()
    types = None if a.types.strip().lower() == "all" else [t.strip() for t in a.types.split(",")]
    df, rep = clean(pd.read_csv(a.inp), property_types=types, min_city_rows=a.min_city_rows,
                     content_dedupe=not a.keep_content_duplicates)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    Path(a.report).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
