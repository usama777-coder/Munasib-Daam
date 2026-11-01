"""Features parsed from the listing URL slug (it contains the ad title words).

Example slug: dha_defence_dha_phase_5_badar_commercial_area_apartment_for_sale-49787055-8484-1.html

 * subarea  : "phase 5", "block j", "sector c" or "none" (big societies hide this in the address)
 * flags    : hand-picked keywords (no price words, so no target leakage)
 * is_installment_ad : price is a down payment / instalment, not the full price -> drop
"""
from __future__ import annotations

import re

import pandas as pd

_SLUG = re.compile(r"/property/(.+?)-\d+-\d+-\d+\.html", re.I)

FLAG_PATTERNS = {
    "brand_new": r"\bbrand new\b",
    "corner": r"\bcorner\b",
    "luxury": r"\b(luxury|luxurious|lavish|elegant|designer)\b",
    "furnished": r"\bfurnished\b",
    "park_facing": r"\bpark facing\b|\bfacing park\b",
    "main_road": r"\bmain (road|boulevard|avenue|market)\b|\bboulevard\b",
    "double_storey": r"\b(double|triple|2|3|two|three) (storey|story|storeys|stories|unit)\b",
    "basement": r"\bbasement\b",
    "renovated": r"\brenovat\w*|\breconstruct\w*",
    "leased": r"\bleased?\b",
}
FLAGS = list(FLAG_PATTERNS)
FLAG_LABELS = {"brand_new": "Brand new", "corner": "Corner", "luxury": "Luxury", "furnished": "Furnished",
               "park_facing": "Park facing", "main_road": "On main road / boulevard",
               "double_storey": "Double storey / unit", "basement": "Basement", "renovated": "Renovated",
               "leased": "Leased title (common in Karachi)"}
INSTALLMENT = (r"instal+ment|\binstal\b|down payment|payment plan|\bbooking\b|easy plan|easy payment"
               r"|pay in \d+ (?:years?|months?)|get possession|possession (?:in|after|within) [\d ]+ (?:years?|months?)"
               r"|\d+ years? (?:plan|payment)")

# an ad about a plot of land mislabelled as a house/flat: it mentions a plot and no built-property word
PLOT_WORD = r"\bplot\b"
BUILT_WORDS = (r"\b(house|villa|bungalow|flat|apartment|portion|penthouse|constructed|built|storey|story|floor|"
               r"rooms?|bed|beds|bedroom|bedrooms)\b")


def is_plot_ad(slug: str) -> bool:
    return bool(re.search(PLOT_WORD, slug)) and not re.search(BUILT_WORDS, slug)

_SUBAREA_PATTERNS = [
    (r"\bphase (\d+|[ivx]+)\b", "phase"),
    (r"\bsector ([a-z]\d{0,2}|\d{1,2}[a-z]?)\b", "sector"),
    (r"\bblock ([a-z]\d?)\b", "block"),
    (r"\b([a-z]\d?) block\b", "block"),
]


def slug_text(url) -> str:
    """Lower-case words of the ad slug, or '' when the URL has no recognisable slug."""
    if not isinstance(url, str):
        return ""
    m = _SLUG.search(url)
    return re.sub(r"[\s_\-]+", " ", m.group(1)).strip().lower() if m else ""


def subarea(slug: str) -> str:
    for pat, label in _SUBAREA_PATTERNS:
        m = re.search(pat, slug)
        if m:
            return f"{label} {m.group(1)}"
    return "none"


def slugs_for(df: pd.DataFrame) -> pd.Series:
    if "url" in df.columns:
        return df["url"].map(slug_text)
    return pd.Series("", index=df.index)
