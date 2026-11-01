"""Why does our area parser disagree with the file's own area_sqft?  Run on the RAW 2024 file.

    python -m hpk.diagnose_area --inp data/raw/zameen.csv
Prints mismatch rate by unit word and example rows. Paste the output back.
"""
import argparse
import re

import pandas as pd

from .ingest import read_any
from .parsing import parse_area_sqft


def diagnose(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    d = df[["area", "area_sqft", "city", "type"]].copy()
    d["mine"] = d["area"].map(parse_area_sqft)
    d["ratio"] = d["mine"] / d["area_sqft"]
    d["unit"] = d["area"].astype(str).str.replace(r"[\d.,\s]+", " ", regex=True).str.strip().str.lower()
    d["agree"] = (d["ratio"] - 1).abs() < 0.02
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True)
    a = ap.parse_args()
    d = diagnose(read_any(a.inp))
    print("overall agreement: %.1f%%   unparseable by us: %d" % (d.agree.mean() * 100, d.mine.isna().sum()))
    by = d.groupby("unit").agg(rows=("agree", "size"), agree_pct=("agree", lambda s: s.mean() * 100),
                               median_ratio=("ratio", "median")).sort_values("rows", ascending=False)
    print("\nBy unit word:\n", by.head(15).round(3).to_string())
    by_c = d.groupby("city").agg(rows=("agree", "size"), agree_pct=("agree", lambda s: s.mean() * 100)).sort_values("rows", ascending=False)
    print("\nBy city:\n", by_c.head(10).round(1).to_string())
    by_t = d.groupby("type").agg(rows=("agree", "size"), agree_pct=("agree", lambda s: s.mean() * 100)).sort_values("rows", ascending=False)
    print("\nBy property type:\n", by_t.head(10).round(1).to_string())
    print("\nExamples of disagreement:\n", d[~d.agree].sample(min(15, (~d.agree).sum()), random_state=0)
          [["area", "area_sqft", "mine", "ratio", "city", "type"]].round(2).to_string())


if __name__ == "__main__":
    main()
