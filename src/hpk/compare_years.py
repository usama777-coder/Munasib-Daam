"""Compare two periods (e.g. 2019 vs 2024): how much did price per sq ft change, by city and area?

    python -m hpk.compare_years --old data/processed/clean_2019.csv --new data/processed/clean_2024.csv
Writes reports/price_growth_by_city.csv and reports/price_growth_by_area.csv.
Caveat for your README: different scrapes, different listing mixes, asking prices only.
"""
import argparse
from pathlib import Path

import pandas as pd


def growth(old: pd.DataFrame, new: pd.DataFrame, by: list[str], min_n: int = 30) -> pd.DataFrame:
    f = lambda d, tag: d.groupby(by).agg(**{f"n_{tag}": ("price_per_sqft", "size"),
                                            f"ppsf_{tag}": ("price_per_sqft", "median")})
    m = f(old, "old").join(f(new, "new"), how="inner")
    m = m[(m.n_old >= min_n) & (m.n_new >= min_n)].copy()
    m["growth_pct"] = (m.ppsf_new / m.ppsf_old - 1) * 100
    return m.sort_values("growth_pct", ascending=False).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    old, new = pd.read_csv(a.old), pd.read_csv(a.new)
    out = Path(a.reports_dir); out.mkdir(exist_ok=True, parents=True)
    city = growth(old, new, ["city"], 100); area = growth(old, new, ["city", "location"])
    city.to_csv(out / "price_growth_by_city.csv", index=False); area.to_csv(out / "price_growth_by_area.csv", index=False)
    print(city.round(1).to_string(index=False)); print("\nTop 10 areas:\n", area.head(10).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
