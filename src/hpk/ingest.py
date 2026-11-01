"""Map downloaded property files to the raw schema the cleaner expects.

Two layouts are recognised automatically:
  A) "zameen2019": property_id; page_url; property_type; price; location; city; baths; area; purpose; ...
  B) "zameen2024": url, title, type, price, area, city, address, bedrooms, baths, area_sqft, ...

    python -m hpk.ingest --inp data/raw/property.csv --out data/raw/listings_2019.csv
    python -m hpk.ingest --inp data/raw/zameen.csv   --out data/raw/listings_2024.csv

IMPORTANT: do NOT pool files from different years into one model, prices have moved a lot.
Train on one period; compare periods with `python -m hpk.compare_years`.
Agent/agency names and fake city-level coordinates are NOT carried over.
"""
import argparse
import re

import pandas as pd

from .parsing import parse_area_sqft

REQUIRED = ["city", "location", "property_type", "price_raw", "area_raw", "bedrooms_raw", "bathrooms_raw"]
KEEP = REQUIRED + ["latitude", "longitude", "listing_date", "url", "source"]
RENT = re.compile(r"for[_\s-]*rent|to[_\s-]*let", re.I)


def read_any(path: str) -> pd.DataFrame:
    """Auto-detects ; , or tab separators and BOMs."""
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def detect_layout(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if {"page_url", "purpose"} <= cols:
        return "zameen2019"
    if {"address", "type", "area_sqft"} <= cols:
        return "zameen2024"
    raise SystemExit(f"Unknown layout. Columns: {list(df.columns)}. Add a mapping in ingest.py")


def _layout_2019(df):
    n0 = len(df)
    df = df[df["purpose"].astype(str).str.strip().str.lower().isin(["for sale", "sale", "buy"])]
    print(f"kept {len(df):,} 'For Sale' rows of {n0:,}")
    return pd.DataFrame({
        "city": df["city"], "location": df["location"], "property_type": df["property_type"],
        "price_raw": df["price"], "area_raw": df["area"], "bedrooms_raw": df["bedrooms"], "bathrooms_raw": df["baths"],
        "latitude": df["latitude"], "longitude": df["longitude"],
        "listing_date": pd.to_datetime(df["date_added"], format="%m-%d-%Y", errors="coerce"), "url": df["page_url"]})


def _layout_2024(df):
    df = df.rename(columns={c: c.lower() for c in df.columns})
    n0 = len(df)
    is_rent = df["url"].astype(str).str.contains(RENT) | df["title"].astype(str).str.contains(RENT)
    df = df[~is_rent]
    print(f"dropped {n0 - len(df):,} rows that look like rentals ('for rent' in url/title); kept {len(df):,}")
    # sanity check: does our area parser agree with the file's own area_sqft column?
    chk = df.sample(min(5000, len(df)), random_state=0)
    mine = chk["area"].map(parse_area_sqft)
    ok = ((mine - chk["area_sqft"]).abs() / chk["area_sqft"] < 0.02)
    print(f"area parser agrees with file's area_sqft on {ok.mean() * 100:.1f}% of a 5,000-row sample")
    return pd.DataFrame({
        "city": df["city"], "location": df["address"].astype(str).str.split(",").str[0].str.strip(),
        "property_type": df["type"], "price_raw": df["price"], "area_raw": df["area"],
        "bedrooms_raw": df["bedrooms"], "bathrooms_raw": df["baths"],
        # coordinates in this file are city-level placeholders (identical for many areas), so dropped
        "listing_date": pd.to_datetime(df["date_added"], errors="coerce"), "url": df["url"]})


def ingest(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    layout = detect_layout(df)
    print("layout:", layout)
    out = (_layout_2019 if layout == "zameen2019" else _layout_2024)(df).copy()
    out["source"] = source or layout
    return out[[c for c in KEEP if c in out.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", default="data/raw/listings.csv")
    ap.add_argument("--source", default="")
    a = ap.parse_args()
    df = ingest(read_any(a.inp), a.source)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, f"{len(df):,} rows; dates {df.listing_date.min()} to {df.listing_date.max()}")
    print(df["city"].value_counts().head(12).to_string())
    print(df["property_type"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
