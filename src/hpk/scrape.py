"""Polite scraper SKELETON. Run it on YOUR machine, not in a sandbox.

Before you fill this in:
  1. Read the site's Terms of Service and /robots.txt. If scraping is not allowed, do NOT scrape;
     use a public dataset instead (see ingest.py) and say so in your README.
  2. Open a listing page in your browser -> right click -> Inspect, and find the CSS selectors
     (or the embedded JSON) for each field. They are deliberately left as TODO because I cannot
     see the live site from here and will not invent selectors.
  3. Keep DELAY_S >= 3, identify yourself in USER_AGENT, cache every page, and never publish
     the raw scraped data (publish aggregates and your code only).

    python -m hpk.scrape --city Islamabad --pages 20
"""
from __future__ import annotations

import argparse
import hashlib
import time
import urllib.robotparser as rp
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://TODO-listing-site.example"          # TODO
LIST_URL = BASE_URL + "/TODO/{city}/?page={page}"        # TODO
USER_AGENT = "student-research-bot (contact: you@example.com)"  # TODO: your real email
DELAY_S = 4.0
CACHE = Path("data/raw/html_cache")

SELECTORS = {  # TODO: fill from the page you inspected
    "card": "TODO",
    "price_raw": "TODO", "area_raw": "TODO", "location": "TODO", "property_type": "TODO",
    "bedrooms_raw": "TODO", "bathrooms_raw": "TODO", "url": "TODO",
}


def allowed(url: str) -> bool:
    p = urlparse(url)
    r = rp.RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    r.read()
    return r.can_fetch(USER_AGENT, url)


def fetch(url: str, session: requests.Session) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".html")
    if f.exists():
        return f.read_text(encoding="utf-8")
    if not allowed(url):
        print("robots.txt disallows", url); return None
    time.sleep(DELAY_S)
    r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    if r.status_code != 200:
        print("HTTP", r.status_code, url); return None
    f.write_text(r.text, encoding="utf-8")
    return r.text


def parse_page(html: str, city: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select(SELECTORS["card"]):
        def txt(key):
            el = card.select_one(SELECTORS[key])
            return el.get_text(" ", strip=True) if el else None
        link = card.select_one(SELECTORS["url"])
        rows.append({"city": city, **{k: txt(k) for k in SELECTORS if k not in ("card", "url")},
                     "url": BASE_URL + link["href"] if link and link.get("href") else None,
                     "listing_date": pd.Timestamp.today().normalize()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--out", default="data/raw/listings.csv")
    a = ap.parse_args()
    if "TODO" in BASE_URL:
        raise SystemExit("Fill in BASE_URL / LIST_URL / SELECTORS first (see the docstring).")
    s, rows = requests.Session(), []
    for page in range(1, a.pages + 1):
        html = fetch(LIST_URL.format(city=a.city.lower(), page=page), s)
        if not html:
            break
        got = parse_page(html, a.city)
        if not got:
            break
        rows += got
        print(a.city, "page", page, len(got), "listings")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out.exists():
        new = pd.concat([pd.read_csv(out), new]).drop_duplicates(subset="url")
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
