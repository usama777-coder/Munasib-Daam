"""Parsers for messy Pakistani real-estate strings.

Prices:  "1.5 Crore", "85 Lakh", "1 Crore 20 Lakh", "2.1 Arab", "PKR 45,00,000"
Areas:   "5 Marla", "1 Kanal", "10 Marla 2 Sarsai", "200 Sq. Yd.", "1,800 sqft"

All functions return None when the text cannot be parsed, never raise.
"""
from __future__ import annotations

import re

# ---- price units (PKR) ------------------------------------------------------
PRICE_UNITS = {
    "arab": 1_000_000_000,
    "arb": 1_000_000_000,
    "crore": 10_000_000,
    "cr": 10_000_000,
    "lakh": 100_000,
    "lac": 100_000,
    "lk": 100_000,
    "thousand": 1_000,
    "k": 1_000,
}

# ---- area units (square feet) ----------------------------------------------
# NOTE: the marla is not standardised. 272.25 sqft is the revenue-record marla
# used in Punjab and by most portals. Some societies/CDA use 225 sqft.
# Change MARLA_SQFT once, here, if you decide to follow a different convention
# and write the choice in your README.
MARLA_SQFT = 272.25
AREA_UNITS = {
    "sarsai": MARLA_SQFT / 9,         # 1 marla = 9 sarsai
    "marla": MARLA_SQFT,
    "kanal": 20 * MARLA_SQFT,         # 1 kanal = 20 marla
    "acre": 43_560.0,
    "sqyd": 9.0,
    "sqm": 10.7639,
    "sqft": 1.0,
}
_AREA_ALIASES = {
    "marla": "marla", "marlas": "marla", "marly": "marla",
    "kanal": "kanal", "kanals": "kanal", "kanaal": "kanal",
    "sarsai": "sarsai", "sarsahi": "sarsai",
    "acre": "acre", "acres": "acre",
    "sqyd": "sqyd", "sqyds": "sqyd", "sqyard": "sqyd", "sqyards": "sqyd",
    "squareyard": "sqyd", "squareyards": "sqyd", "yd": "sqyd", "yard": "sqyd", "yards": "sqyd",
    "sqm": "sqm", "m2": "sqm", "squaremeter": "sqm", "squaremeters": "sqm",
    "squaremetre": "sqm", "squaremetres": "sqm",
    "sqft": "sqft", "sqfeet": "sqft", "sqfoot": "sqft", "squarefeet": "sqft",
    "squarefoot": "sqft", "sft": "sqft", "ft2": "sqft",
}

_NUM = r"\d+(?:,\d+)*(?:\.\d+)?"
_NO_PRICE_WORDS = ("call", "contact", "ask", "negotiable", "tbd", "n/a", "na")


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_price(text) -> float | None:
    """Return a price in PKR, or None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text) if text == text and text > 0 else None
    s = str(text).strip().lower()
    if not s or any(w in s for w in _NO_PRICE_WORDS):
        return None
    s = re.sub(r"\b(pkr|rs\.?|rupees?)\b", " ", s)
    parts = re.findall(rf"({_NUM})\s*([a-z]+)?", s)
    if not parts:
        return None
    total, matched_unit = 0.0, False
    for num, unit in parts:
        value = _to_float(num)
        if unit and unit in PRICE_UNITS:
            total += value * PRICE_UNITS[unit]
            matched_unit = True
        elif unit:
            continue  # unknown word after a number, ignore
        elif not matched_unit:
            total += value  # bare number, assume PKR
    return total if total > 0 else None


def _canon_unit(word: str) -> str | None:
    w = re.sub(r"[^a-z0-9]", "", word.lower())
    return _AREA_ALIASES.get(w)


def parse_area_sqft(text) -> float | None:
    """Return area in square feet, or None. Handles compound areas like '1 Kanal 5 Marla'."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text) if text == text and text > 0 else None
    s = str(text).strip().lower().replace("sq.", "sq").replace("sq ", "sq")
    s = s.replace("square ", "square")
    pairs = re.findall(rf"({_NUM})\s*([a-z][a-z0-9.\s]*?)(?=\s*{_NUM}|\s*$|,)", s)
    total = 0.0
    for num, word in pairs:
        unit = _canon_unit(word.strip())
        if unit:
            total += _to_float(num) * AREA_UNITS[unit]
    return total if total > 0 else None


def parse_count(text) -> int | None:
    """Bedrooms/bathrooms: '5', '5 Beds', 'Studio' -> 0."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text) if text == text and text >= 0 else None
    s = str(text).lower()
    if "studio" in s:
        return 0
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None
