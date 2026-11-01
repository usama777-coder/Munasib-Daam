"""PLACEHOLDER DATA GENERATOR - FOR TESTING THE PIPELINE ONLY.

This produces fake listings in the same messy formats real portals use. It exists so
the pipeline, tests and app can run before you have scraped anything. Never report
metrics from this data on your CV or in your README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parsing import MARLA_SQFT

CITIES = {
    # city: (price per sqft base in PKR, [areas], lat, lon)
    "Islamabad": (14_000, ["DHA Phase 2", "Bahria Town", "G-13", "F-11", "I-8", "Gulberg Greens"], 33.68, 73.05),
    "Lahore": (12_000, ["DHA Phase 5", "Bahria Town", "Johar Town", "Gulberg", "Model Town", "Wapda Town"], 31.52, 74.35),
    "Karachi": (11_000, ["DHA Phase 6", "Gulshan-e-Iqbal", "North Nazimabad", "Bahria Town", "Scheme 33"], 24.86, 67.01),
    "Peshawar": (7_500, ["Hayatabad", "University Town", "Regi Model Town", "Warsak Road", "Gulbahar"], 34.01, 71.52),
}


def _fmt_price(p: float, rng) -> str:
    if p >= 1e7:
        return f"{p / 1e7:.2f} Crore".replace(".00 ", " ")
    if rng.random() < 0.15:
        return f"PKR {int(p):,}"
    return f"{p / 1e5:.0f} Lakh"


def _fmt_area(sqft: float, city: str, rng) -> str:
    if city == "Karachi" and rng.random() < 0.5:
        return f"{round(sqft / 9)} Sq. Yd."
    marla = sqft / MARLA_SQFT
    if marla >= 20:
        return f"{marla / 20:.1f} Kanal"
    return f"{round(marla)} Marla"


def make_synthetic(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        city = rng.choice(list(CITIES))
        base, areas, lat, lon = CITIES[city]
        area_name = rng.choice(areas)
        area_mult = 0.75 + 0.15 * areas.index(area_name)
        marla = float(rng.choice([3, 5, 7, 10, 10, 15, 20, 40]))
        sqft = marla * MARLA_SQFT
        beds = int(np.clip(round(marla / 2.2 + rng.normal(0, 0.7)), 1, 9))
        baths = int(np.clip(beds + rng.integers(-1, 2), 1, 9))
        ptype = rng.choice(["House", "House", "House", "Flat", "Upper Portion"], p=[.4, .2, .1, .2, .1])
        if ptype == "Flat":
            sqft = max(450, sqft * 0.35)
            beds = min(beds, 4)
        age_mult = rng.choice([1.0, 0.95, 0.85])
        ppsf = base * area_mult * age_mult * np.exp(rng.normal(0, 0.18)) * (sqft / 2000) ** -0.12
        price = ppsf * sqft
        rows.append({
            "city": city, "location": area_name, "property_type": ptype,
            "price_raw": _fmt_price(price, rng), "area_raw": _fmt_area(sqft, city, rng),
            "bedrooms_raw": str(beds) if rng.random() > 0.03 else "-", "bathrooms_raw": str(baths),
            "latitude": lat + rng.normal(0, 0.03), "longitude": lon + rng.normal(0, 0.03),
            "listing_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 270))),
            "url": f"https://example.invalid/listing/{rng.integers(1e9)}",
        })
    df = pd.DataFrame(rows)
    # add realistic junk: exact duplicates and unparseable prices
    df = pd.concat([df, df.sample(frac=0.03, random_state=seed)], ignore_index=True)
    df.loc[df.sample(frac=0.02, random_state=seed + 1).index, "price_raw"] = "Call for price"
    return df


if __name__ == "__main__":
    out = "data/sample/SYNTHETIC_listings_raw.csv"
    make_synthetic().to_csv(out, index=False)
    print("wrote", out, "(placeholder data, do not report results from it)")
