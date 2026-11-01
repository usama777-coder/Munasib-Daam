"""Pure helpers for the Streamlit app: formatting, verdicts and small HTML widgets.

Kept free of Streamlit so it can be unit-tested. All text that comes from data is escaped.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from .textfeats import FLAG_LABELS

DRIVER_LABELS = {"city": "City", "area": "Area or society", "area_sub": "Phase, block or sector",
                 "property_type": "Property type", "log_area": "Size", "bedrooms": "Bedrooms",
                 "bathrooms": "Bathrooms", **FLAG_LABELS}
UNITS = ["Marla", "Kanal", "Sq. Yd.", "Sq. Ft."]
PROPERTY_TYPES = ["House", "Flat", "Upper Portion", "Lower Portion", "Penthouse"]

APP_NAME = "Munasib Daam"  # "a fair price" in Urdu. Change this one line to rename the whole app.
TAGLINE = "Ghar ka sahi daam. Know the fair price before you negotiate."

# Colour themes. hero-a/b = header gradient, s1..s4 = the stripe band and tile accents.
THEMES = {
    "Truck art": dict(hero_a="#c2185b", hero_b="#5b2a9e", primary="#d81b60", accent="#f59e0b",
                      good="#2e9e5b", bad="#e0483e", s1="#f6b800", s2="#12b5b0", s3="#e53935", s4="#43a047", bg_a="#ffd3e6", bg_b="#e4d3ff", bg_c="#c6f1ed", bg_d="#fff0b8"),
    "Indus": dict(hero_a="#1e3a8a", hero_b="#3b63c9", primary="#3b63c9", accent="#e8a33d",
                  good="#2f9e63", bad="#d4544a", s1="#e8a33d", s2="#0fa3a3", s3="#2a7fba", s4="#7c4dff", bg_a="#d3e2ff", bg_b="#cdeff2", bg_c="#e3dcff", bg_d="#ffeccc"),
    "Sabz": dict(hero_a="#0b6b4b", hero_b="#0e7490", primary="#138a5c", accent="#f4b400",
                 good="#2f9e63", bad="#d4544a", s1="#f4b400", s2="#138a5c", s3="#0e7490", s4="#c2410c", bg_a="#d0f4e2", bg_b="#d3eff5", bg_c="#e6f6c8", bg_d="#fff1bf"),
    "Sunset": dict(hero_a="#c2410c", hero_b="#9d174d", primary="#e4572e", accent="#f59e0b",
                   good="#2f9e63", bad="#c62863", s1="#f59e0b", s2="#e4572e", s3="#9d174d", s4="#7c3aed", bg_a="#ffddc9", bg_b="#ffcfe0", bg_c="#e4d5ff", bg_d="#ffefc2"),
}
DEFAULT_THEME = "Truck art"
TONE_ICON = {"good": "🎉", "calm": "👍", "warn": "⚠️", "bad": "🚩"}


def theme_css(name: str) -> str:
    t = THEMES.get(name) or THEMES[DEFAULT_THEME]
    return "<style>:root{" + ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in t.items()) + "}</style>"


def pkr_words(x: float) -> str:
    """Spoken Pakistani style: '2 crore 59 lakh'."""
    lakh_total = int(round(float(x) / 1e5))
    crore, lakh = divmod(lakh_total, 100)
    parts = ([f"{crore} crore"] if crore else []) + ([f"{lakh} lakh"] if lakh else [])
    return " ".join(parts) or f"{float(x):,.0f} rupees"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def pkr(x: float) -> str:
    x = float(x)
    if x >= 1e7:
        return f"Rs {x / 1e7:.2f} crore"
    if x >= 1e5:
        return f"Rs {x / 1e5:.1f} lakh"
    return f"Rs {x:,.0f}"


# ---------------------------------------------------------------- estimate
def predict(model, meta: dict, row: pd.DataFrame, features: list[str]):
    """Return (estimate, low, high) in PKR. The range is the 10th to 90th percentile of out-of-fold errors."""
    est = float(np.exp(model.predict(row[features])[0]))
    lo_log, hi_log = meta["interval_log"]
    return est, est * float(np.exp(lo_log)), est * float(np.exp(hi_log))


def deal_verdict(asking: float, est: float) -> dict:
    """Compare an asking price with the estimate. Bands are wider than the model's typical error (~14%)."""
    ratio = asking / est
    pct = abs(ratio - 1) * 100
    if ratio < 0.70:
        return dict(tone="good", label="Far below the estimate", ratio=ratio,
                    text=f"{pct:.0f}% below the estimate. Check why before getting excited: unfinished work, "
                         "lease title, a low floor or a payment-plan price can all explain a gap this size.")
    if ratio < 0.85:
        return dict(tone="good", label="Below the estimate", ratio=ratio, text=f"{pct:.0f}% below the estimate.")
    if ratio <= 1.15:
        return dict(tone="calm", label="In line with the estimate", ratio=ratio,
                    text="Within the model's usual margin of error.")
    if ratio <= 1.40:
        return dict(tone="warn", label="Above the estimate", ratio=ratio, text=f"{pct:.0f}% above the estimate.")
    return dict(tone="bad", label="Far above the estimate", ratio=ratio,
                text=f"{pct:.0f}% above the estimate. Ask what the seller is charging for that similar listings are not.")


def driver_rows(contrib: pd.Series, top: int = 8, min_abs_pct: float = 0.5,
                details: dict | None = None) -> list[tuple[str, float]]:
    """SHAP log-price contributions -> [(label, approx % effect vs the average listing)], largest first.
    details: optional {feature: "5 marla"} so the label reads "Size (5 marla)"."""
    pct = (np.exp(contrib.astype(float)) - 1) * 100
    pct = pct[pct.abs() >= min_abs_pct]
    pct = pct.reindex(pct.abs().sort_values(ascending=False).index).head(top)
    details = details or {}

    def label(k):
        base = DRIVER_LABELS.get(k, k)
        return f"{base} ({details[k]})" if details.get(k) else base
    return [(label(k), float(v)) for k, v in pct.items()]


# ---------------------------------------------------------------- area stats
def area_context(stats: pd.DataFrame, city: str, location: str) -> dict | None:
    if stats is None or stats.empty:
        return None
    m = stats[(stats.city == city) & (stats.location == location)]
    if m.empty:
        return None
    r = m.iloc[0]
    return {"listings": int(r.listings), "median_ppsf": float(r.median_price_per_sqft)}


def popular_area(stats: pd.DataFrame, areas_by_city: dict, city: str) -> str:
    """The area with the most listings in a city, falling back to the first known area."""
    areas = areas_by_city.get(city, [])
    if stats is not None and not stats.empty:
        s = stats[(stats.city == city) & (stats.location.isin(areas))]
        if len(s):
            return str(s.sort_values("listings", ascending=False).iloc[0].location)
    return areas[0] if areas else ""


def city_summary(stats: pd.DataFrame) -> pd.DataFrame:
    """Typical price per sq ft per city: listing-weighted average of area medians (an approximation)."""
    g = stats.groupby("city").apply(
        lambda d: pd.Series({"listings": int(d.listings.sum()),
                             "typical_ppsf": float(np.average(d.median_price_per_sqft, weights=d.listings))}),
        include_groups=False)
    return g.sort_values("typical_ppsf", ascending=False).reset_index()


def metrics_table(meta: dict) -> pd.DataFrame:
    rows = []
    for name, m in meta.get("metrics", {}).items():
        rows.append({"Model": name, "R²": round(m["R2"], 3), "Median error (%)": round(m["MedianAPE_pct"], 1),
                     "Average error (Rs lakh)": round(m["MAE_pkr"] / 1e5, 1)})
    return pd.DataFrame(rows).sort_values("Median error (%)") if rows else pd.DataFrame()


def format_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Readable table for the best-value tab."""
    out = pd.DataFrame({
        "City": df["city"], "Area": df["location"], "Type": df["property_type"],
        "Size (sq ft)": df["area_sqft"].round(0).astype(int), "Beds": df["bedrooms"],
        "Listed price": df["price_pkr"].map(pkr), "Model estimate": df["fair_price_pkr"].map(pkr),
        "Below estimate (%)": ((1 - df["price_ratio"]) * 100).round(0).astype(int),
    })
    if "url" in df.columns:
        out["Ad"] = df["url"]
    return out


# ---------------------------------------------------------------- HTML widgets
def badge_html(label: str, tone: str = "calm") -> str:
    return f'<span class="badge {esc(tone)}">{esc(label)}</span>'


def ruler_html(est: float, lo: float, hi: float, asking: float | None = None) -> str:
    """Horizontal price ruler: likely range as a band, the estimate as a blue marker, asking price as apricot."""
    pts = [lo, hi, est] + ([asking] if asking else [])
    a, b = min(pts), max(pts)
    pad = (b - a) * 0.15 or a * 0.1
    a, b = max(a - pad, 0), b + pad
    pos = lambda v: min(max((v - a) / (b - a) * 100, 2), 98)
    lo_p, hi_p, est_p = pos(lo), pos(hi), pos(est)
    ask = ""
    if asking:
        ask = (f'<div class="rmark ask" style="left:{pos(asking):.1f}%">'
               f'<span class="rlab below">Asking {esc(pkr(asking))}</span></div>')
    return (
        '<div class="ruler">'
        '<div class="rtrack">'
        f'<div class="rband" style="left:{lo_p:.1f}%;width:{max(hi_p - lo_p, 1):.1f}%"></div>'
        f'<div class="rmark est" style="left:{est_p:.1f}%"><span class="rlab">Estimate</span></div>'
        f"{ask}"
        "</div>"
        '<div class="rends">'
        f'<span style="left:{lo_p:.1f}%">{esc(pkr(lo))}</span><span style="left:{hi_p:.1f}%">{esc(pkr(hi))}</span>'
        "</div></div>"
    )


def bars_html(rows: list[tuple[str, float]], signed: bool = False, fmt=None, empty: str = "") -> str:
    """Horizontal bars. signed=True diverges from a centre line (green up, red down); otherwise bars grow from the left."""
    if not rows:
        return f'<p class="muted">{esc(empty)}</p>'
    fmt = fmt or (lambda v: f"{v:+.0f}%" if signed else f"{v:,.0f}")
    top = max(abs(v) for _, v in rows) or 1.0
    out = ['<div class="bars">']
    for label, v in rows:
        w = abs(v) / top * (50 if signed else 100)
        if signed:
            side = "right:50%" if v < 0 else "left:50%"
            cls = "neg" if v < 0 else "pos"
            fill = f'<div class="bfill {cls}" style="{side};width:{w:.1f}%"></div><div class="bmid"></div>'
        else:
            fill = f'<div class="bfill flat" style="left:0;width:{w:.1f}%"></div>'
        out.append(f'<div class="brow"><div class="blab">{esc(label)}</div>'
                   f'<div class="btrack">{fill}</div><div class="bval">{esc(fmt(v))}</div></div>')
    out.append("</div>")
    return "".join(out)


def hero_html() -> str:
    """The coloured title band only. The tagline and description live in intro_html, below it."""
    return (f'<div class="hero"><div class="hero-in"><div class="brand">{esc(APP_NAME)}</div></div>'
            '<div class="stripe"></div></div>')


def intro_html(n_listings: int, n_cities: int, typical_err: float | None) -> str:
    err = f" Estimates are typically within about {typical_err:.0f}% of the asking price." if typical_err else ""
    return (
        '<div class="intro">'
        f'<div class="tagline">{esc(TAGLINE)}</div>'
        f'<p class="intro-p">Price estimates for houses, flats and portions in {n_cities} Pakistani cities, learned from '
        f'{n_listings:,} real listings from May 2024. These are asking prices, not final sale prices.{esc(err)}</p>'
        '</div>')


def tile_html(label: str, value: str, sub: str = "", color: str = "s1") -> str:
    return (f'<div class="tile" style="--tc:var(--{esc(color)})"><div class="tl">{esc(label)}</div>'
            f'<div class="tv">{esc(value)}</div><div class="ts">{esc(sub)}</div></div>')


def tiles_html(tiles: list[str]) -> str:
    return '<div class="tiles">' + "".join(tiles) + "</div>"
