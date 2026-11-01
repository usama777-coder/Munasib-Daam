"""Munasib Daam: house price estimates for Pakistan.   Run:  streamlit run app/streamlit_app.py"""
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hpk import appcore as ac  # noqa: E402
from hpk.explain import shap_with_base  # noqa: E402
from hpk.modeling import FEATURES, add_features  # noqa: E402
from hpk.parsing import MARLA_SQFT, parse_area_sqft  # noqa: E402
from hpk.textfeats import FLAG_LABELS  # noqa: E402

st.set_page_config(page_title=f"{ac.APP_NAME} | {ac.TAGLINE}", page_icon="🏡", layout="wide")

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&display=swap');
:root { --hero-a:#c2185b; --hero-b:#5b2a9e; --primary:#d81b60; --accent:#f59e0b; --good:#2e9e5b; --bad:#e0483e;
        --s1:#f6b800; --s2:#12b5b0; --s3:#e53935; --s4:#43a047;
        --bg-a:#ffd3e6; --bg-b:#e4d3ff; --bg-c:#c6f1ed; --bg-d:#fff0b8;
        --line:rgba(128,128,128,.28); --soft:rgba(128,128,128,.10); }
footer { visibility: hidden; }
.stApp { background: radial-gradient(circle, rgba(255,255,255,.6) 1.6px, transparent 2.1px) 0 0/26px 26px,
                     linear-gradient(135deg, var(--bg-a) 0%, var(--bg-b) 35%, var(--bg-c) 70%, var(--bg-d) 100%);
         background-attachment: fixed; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], header[data-testid="stHeader"] { background: transparent; }
.stApp::before { content:""; position:fixed; top:0; left:0; right:0; height:8px; z-index:1000;
                 background: repeating-linear-gradient(90deg, var(--s1) 0 28px, var(--s2) 28px 56px, var(--s3) 56px 84px, var(--s4) 84px 112px); }
.block-container { max-width:1180px; margin:1.6rem auto 2rem auto; padding:1.6rem 2.2rem 2.2rem 2.2rem !important;
                   background:rgba(255,255,255,.84); border-radius:24px; border:1px solid rgba(255,255,255,.9);
                   box-shadow:0 14px 48px rgba(70,30,110,.16); }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; }
.stTabs [data-baseweb="tab"] { border-radius:12px 12px 0 0; padding:.5rem 1rem; font-weight:600; }
.stTabs [aria-selected="true"] { color:var(--primary) !important; background:color-mix(in srgb, var(--primary) 11%, transparent); }
.stTabs [data-baseweb="tab-highlight"] { background-color:var(--primary) !important; height:3px; }
.stButton > button { border:2px solid var(--primary); border-radius:14px; font-weight:600; transition:all .15s ease; }
.stButton > button:hover { background:var(--primary); color:#fff; border-color:var(--primary); }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:rgba(255,255,255,.55); }
div[data-baseweb="select"] > div, div[data-baseweb="base-input"], div[data-baseweb="input"] {
    background-color:rgba(128,128,128,.07) !important; background-color:color-mix(in srgb, var(--primary) 8%, white) !important;
    border-color:color-mix(in srgb, var(--primary) 30%, transparent) !important; border-radius:10px !important; }
.hero { position:relative; border-radius:18px; overflow:hidden; margin:0;
        background: radial-gradient(circle, rgba(255,255,255,.17) 2px, transparent 2.6px) 0 0/22px 22px,
                    linear-gradient(120deg, var(--hero-a), var(--hero-b)); }
.hero, .hero * { color:#fff !important; }
.hero-in { padding:1.5rem 2rem 1.3rem 2rem; }
.brand { font-family:'Bricolage Grotesque', system-ui, sans-serif; font-weight:700; font-size:3.4rem;
         line-height:1; letter-spacing:-.025em; }
.intro { margin:1rem 0 1.6rem 0; }
.tagline { font-family:'Bricolage Grotesque', system-ui, sans-serif; font-size:1.35rem; font-weight:700; margin:0 0 .35rem 0; }
.intro-p { max-width:66ch; opacity:.82; margin:0; font-size:1rem; }
.stripe { height:11px; background: repeating-linear-gradient(90deg, var(--s1) 0 28px, var(--s2) 28px 56px,
          var(--s3) 56px 84px, var(--s4) 84px 112px); }
.muted { opacity:.72; font-size:.93rem; }
.h3 { font-family:'Bricolage Grotesque', system-ui, sans-serif; font-weight:700; font-size:1.3rem; margin:1.7rem 0 .35rem 0;
      border-left:6px solid var(--primary); padding-left:.7rem; }
.result { border-left:6px solid var(--primary); border-radius:6px; padding:.5rem 0 .6rem 1.5rem; margin-bottom:1.2rem;
          background: linear-gradient(90deg, rgba(128,128,128,.07), transparent 70%); }
.result.tone-good { border-left-color:var(--good); } .result.tone-warn { border-left-color:var(--accent); }
.result.tone-bad { border-left-color:var(--bad); } .result.tone-calm { border-left-color:var(--primary); }
.rlabel { font-size:.95rem; opacity:.75; }
.rbig { font-family:'Bricolage Grotesque', system-ui, sans-serif; font-weight:700; font-size:3.3rem; line-height:1.05;
        letter-spacing:-.02em; font-variant-numeric:tabular-nums; margin:.1rem 0 .1rem 0; color:var(--primary); }
.rwords { font-size:1rem; opacity:.75; margin-bottom:.4rem; }
.rsub { opacity:.82; font-size:.98rem; max-width:54ch; }
.ruler { position:relative; margin:2.8rem .4rem 2.5rem .4rem; }
.rtrack { position:relative; height:12px; border-radius:7px; background:var(--soft); border:1px solid var(--line); }
.rband { position:absolute; top:-1px; bottom:-1px; border-radius:7px; background:var(--primary); opacity:.4; }
.rmark { position:absolute; top:-7px; width:24px; height:24px; margin-left:-12px; border-radius:50%;
         border:3px solid #fff; box-shadow:0 1px 6px rgba(0,0,0,.4); }
.rmark.est { background:var(--primary); } .rmark.ask { background:var(--accent); }
.rlab { position:absolute; bottom:32px; left:50%; transform:translateX(-50%); white-space:nowrap; font-size:.82rem; font-weight:700; }
.rlab.below { bottom:auto; top:32px; color:var(--accent); }
.rends { position:relative; height:1.2rem; margin-top:.6rem; font-size:.8rem; opacity:.78; }
.rends span { position:absolute; transform:translateX(-50%); white-space:nowrap; }
.badge { display:inline-block; padding:.32rem .9rem; border-radius:999px; font-weight:700; font-size:.92rem; }
.badge.good { background:rgba(46,158,91,.18); color:var(--good); }
.badge.calm { background:rgba(128,128,128,.16); }
.badge.warn { background:rgba(245,158,11,.2); color:#b87a12; }
.badge.bad  { background:rgba(224,72,62,.17); color:var(--bad); }
.verdict-text { margin:.5rem 0 0 0; opacity:.88; max-width:58ch; }
.tiles { display:grid; grid-template-columns:repeat(3, 1fr); gap:.8rem; margin:.2rem 0 .4rem 0; }
.tile { border-top:5px solid var(--tc); border-radius:12px; padding:.75rem 1rem .8rem 1rem;
        background:rgba(128,128,128,.09); background:color-mix(in srgb, var(--tc) 13%, transparent); }
.tl { font-size:.85rem; opacity:.78; } .ts { font-size:.82rem; opacity:.72; }
.tv { font-family:'Bricolage Grotesque', system-ui, sans-serif; font-weight:700; font-size:1.5rem; line-height:1.25;
      font-variant-numeric:tabular-nums; }
.bars { margin:.5rem 0 .3rem 0; }
.brow { display:grid; grid-template-columns:minmax(8rem, 13rem) 1fr 4.6rem; align-items:center; gap:.8rem; padding:.3rem 0; }
.blab { font-size:.94rem; } .btrack { position:relative; height:16px; }
.bfill { position:absolute; top:2px; bottom:2px; border-radius:4px; }
.bfill.pos { background:var(--good); } .bfill.neg { background:var(--bad); }
.bfill.flat { background:linear-gradient(90deg, var(--s2), var(--primary)); }
.bmid { position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:var(--line); }
.bval { text-align:right; font-variant-numeric:tabular-nums; font-size:.9rem; opacity:.88; }
@media (max-width: 700px) {
  .brand { font-size:2.5rem; } .rbig { font-size:2.4rem; } .tiles { grid-template-columns:1fr; }
  .block-container { margin:.6rem .4rem 1rem .4rem; padding:1rem 1rem 1.6rem 1rem !important; border-radius:18px; }
  .brow { grid-template-columns:7rem 1fr 3.6rem; gap:.5rem; } .hero-in { padding:1.3rem 1.2rem; }
}
</style>
"""


# ------------------------------------------------------------------ data
@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "model.joblib")
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    p = ROOT / "models" / "area_stats.csv"
    stats = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return model, meta, stats


def read_optional(rel: str, kind: str = "csv"):
    p = ROOT / rel
    if not p.exists():
        return None
    return pd.read_csv(p) if kind == "csv" else json.loads(p.read_text())


if not (ROOT / "models" / "model.joblib").exists():
    st.error("No trained model found. Run `python -m hpk.train` first (see the README).")
    st.stop()

model, meta, stats = load()
cities = sorted(meta["areas_by_city"])
n_listings = meta["n_train"] + meta["n_test"]
typical_err = meta.get("metrics", {}).get(meta["model"], {}).get("MedianAPE_pct")
ss = st.session_state

# ------------------------------------------------------------------ theme + header
ss.setdefault("theme", ac.DEFAULT_THEME)
_, theme_col = st.columns([5, 2])
theme_col.selectbox("Colour theme", list(ac.THEMES), key="theme")
st.markdown(BASE_CSS + ac.theme_css(ss["theme"]), unsafe_allow_html=True)
st.markdown(ac.hero_html(), unsafe_allow_html=True)
st.markdown(ac.intro_html(n_listings, len(cities), typical_err), unsafe_allow_html=True)

# ------------------------------------------------------------------ defaults and example buttons
ss.setdefault("city", "Lahore" if "Lahore" in cities else cities[0])
ss.setdefault("ptype", "House")
ss.setdefault("size", 5.0)
ss.setdefault("unit", "Marla")
ss.setdefault("beds", 3)
ss.setdefault("baths", 3)
ss.setdefault("ask", 0.0)
ss.setdefault("sub", "")
ss.setdefault("loc", ac.popular_area(stats, meta["areas_by_city"], ss["city"]))
for k in FLAG_LABELS:
    ss.setdefault(f"flag_{k}", False)


def apply_example(city, ptype, size, unit, beds, baths):
    ss["city"], ss["ptype"], ss["size"], ss["unit"] = city, ptype, float(size), unit
    ss["beds"], ss["baths"], ss["ask"], ss["sub"] = beds, baths, 0.0, ""
    ss["loc"] = ac.popular_area(stats, meta["areas_by_city"], city)


EXAMPLES = [("🏡 5 marla house, Lahore", ("Lahore", "House", 5, "Marla", 3, 3)),
            ("🏛️ 10 marla house, Islamabad", ("Islamabad", "House", 10, "Marla", 4, 4)),
            ("🏢 3-bed flat, Karachi", ("Karachi", "Flat", 1200, "Sq. Ft.", 3, 2)),
            ("🏔️ 5 marla house, Peshawar", ("Peshawar", "House", 5, "Marla", 3, 3))]

tab_names = ["🏠 Estimate", "🗺️ Compare areas"]
best_value = read_optional("reports/undervalued_listings.csv")
if best_value is not None:
    tab_names.append("💎 Best-value listings")
tab_names.append("📊 About the numbers")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ------------------------------------------------------------------ estimate
with tabs["🏠 Estimate"]:
    form, result = st.columns([5, 7], gap="large")

    with form:
        st.caption("Try an example")
        ex_cols = st.columns(2)
        for i, (label, args) in enumerate(EXAMPLES):
            ex_cols[i % 2].button(label, key=f"ex{i}", on_click=apply_example, args=args)

        st.selectbox("City", cities, key="city")
        areas = meta["areas_by_city"][ss["city"]]
        if ss.get("loc") not in areas:
            ss["loc"] = ac.popular_area(stats, meta["areas_by_city"], ss["city"])
        st.selectbox("Area or society", areas, key="loc")
        st.selectbox("Property type", ac.PROPERTY_TYPES, key="ptype")
        c1, c2 = st.columns([1, 1])
        c1.number_input("Size", min_value=1.0, step=1.0, key="size")
        c2.selectbox("Unit", ac.UNITS, key="unit")
        c3, c4 = st.columns(2)
        c3.slider("Bedrooms", 0, 10, key="beds")
        c4.slider("Bathrooms", 1, 10, key="baths")
        with st.expander("More details (optional)"):
            st.text_input("Phase, block or sector", key="sub", placeholder="for example: phase 5, block j, sector c",
                          help="Big societies such as DHA or Bahria Town have very different prices by phase.")
            f_cols = st.columns(2)
            for i, (k, label) in enumerate(FLAG_LABELS.items()):
                f_cols[i % 2].checkbox(label, key=f"flag_{k}")
        st.number_input("Asking price in lakh (optional)", min_value=0.0, step=5.0, key="ask",
                        help="Enter the price in the ad to see where it falls. 100 lakh = 1 crore.")

    sqft = parse_area_sqft(f"{ss['size']} {ss['unit']}")
    row = add_features(pd.DataFrame([{
        "city": ss["city"], "location": ss["loc"], "property_type": ss["ptype"], "area_sqft": sqft,
        "bedrooms": ss["beds"], "bathrooms": ss["baths"], "subarea": ss["sub"].strip().lower() or "none",
        **{k: int(ss[f"flag_{k}"]) for k in FLAG_LABELS}}]))
    est, lo, hi = ac.predict(model, meta, row, FEATURES)
    asking = ss["ask"] * 1e5 if ss["ask"] > 0 else None

    with result:
        tone, verdict_html = "none", ""
        if asking:
            v = ac.deal_verdict(asking, est)
            tone = v["tone"]
            verdict_html = (f'{ac.badge_html(ac.TONE_ICON[tone] + " " + v["label"], tone)}'
                            f'<p class="verdict-text">{ac.esc(v["text"])}</p>')
        st.markdown(
            f'<div class="result tone-{tone}">'
            '<div class="rlabel">Estimated price</div>'
            f'<div class="rbig">{ac.esc(ac.pkr(est))}</div>'
            f'<div class="rwords">That is {ac.esc(ac.pkr_words(est))}</div>'
            f'<div class="rsub">About 8 in 10 comparable listings are priced between '
            f'{ac.esc(ac.pkr(lo))} and {ac.esc(ac.pkr(hi))}.</div>'
            f'{ac.ruler_html(est, lo, hi, asking)}{verdict_html}</div>', unsafe_allow_html=True)

        ctx = ac.area_context(stats, ss["city"], ss["loc"])
        ppsf = est / sqft
        if ctx:
            diff = ppsf / ctx["median_ppsf"] - 1
            area_tile = ac.tile_html("Area median per sq ft", f"Rs {ctx['median_ppsf']:,.0f}",
                                     f"This property: {diff:+.0%} · {ctx['listings']} listings", "s2")
        else:
            area_tile = ac.tile_html("Area median per sq ft", "Not enough listings", "", "s2")
        st.markdown(ac.tiles_html([
            ac.tile_html("Price per sq ft", f"Rs {ppsf:,.0f}", "estimated", "s1"), area_tile,
            ac.tile_html("Size", f"{sqft:,.0f} sq ft", f"about {sqft / MARLA_SQFT:.1f} marla", "s3")]),
            unsafe_allow_html=True)

        st.markdown('<div class="h3">What moves this price</div>', unsafe_allow_html=True)
        shap_res = shap_with_base(model, row[FEATURES])
        if shap_res is None:
            st.markdown('<p class="muted">The price breakdown appears when the SHAP library is installed.</p>',
                        unsafe_allow_html=True)
        else:
            contrib, base_log = shap_res
            details = {"city": ss["city"], "area": ss["loc"], "property_type": ss["ptype"],
                       "log_area": f"{ss['size']:g} {ss['unit'].lower()}", "bedrooms": str(ss["beds"]),
                       "bathrooms": str(ss["baths"]), "area_sub": ss["sub"].strip() or None}
            st.markdown(f'<p class="muted">The average listing in the data is priced around '
                        f'<b>{ac.esc(ac.pkr(math.exp(base_log)))}</b>. Each bar shows how one detail of this property '
                        'moves the price compared with that average. Green pushes it up and red pulls it down, so a '
                        'red bar for size just means smaller than the average listing, and a red bar for city '
                        'means cheaper than the average of all cities.</p>', unsafe_allow_html=True)
            st.markdown(ac.bars_html(ac.driver_rows(contrib.iloc[0], details=details), signed=True,
                                     empty="Nothing here shifts the price much."), unsafe_allow_html=True)
            st.markdown(f'<p class="muted">Effects multiply, so the bars do not add up exactly. '
                        f'Together they take the average listing to your estimate of <b>{ac.esc(ac.pkr(est))}</b>.</p>',
                        unsafe_allow_html=True)
        if typical_err:
            st.markdown(f'<p class="muted">On listings it had never seen, the estimate was typically within '
                        f'{typical_err:.0f}% of the asking price. Use it as a starting point for negotiation, '
                        f'not as a valuation.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ compare areas
with tabs["🗺️ Compare areas"]:
    if stats.empty:
        st.info("Area statistics are not available in this deployment.")
    else:
        st.markdown('<div class="h3">Typical price per sq ft by city</div>', unsafe_allow_html=True)
        cs = ac.city_summary(stats)
        st.markdown(ac.bars_html([(r.city, r.typical_ppsf) for r in cs.itertuples()],
                                 fmt=lambda v: f"Rs {v:,.0f}"), unsafe_allow_html=True)
        st.markdown('<div class="h3">Inside one city</div>', unsafe_allow_html=True)
        pick = st.selectbox("City", cities, key="mkt_city", index=cities.index(ss["city"]))
        s = stats[(stats.city == pick) & (stats.listings >= 10)].sort_values("median_price_per_sqft", ascending=False)
        if s.empty:
            st.markdown('<p class="muted">No area in this city has 10 or more listings.</p>', unsafe_allow_html=True)
        else:
            top, cheap, busy = s.iloc[0], s.iloc[-1], s.sort_values("listings", ascending=False).iloc[0]
            st.markdown(ac.tiles_html([
                ac.tile_html("Priciest area", str(top.location), f"Rs {top.median_price_per_sqft:,.0f} per sq ft", "s3"),
                ac.tile_html("Most affordable area", str(cheap.location), f"Rs {cheap.median_price_per_sqft:,.0f} per sq ft", "s4"),
                ac.tile_html("Most listings", str(busy.location), f"{int(busy.listings)} listings", "s2")]),
                unsafe_allow_html=True)
            st.markdown('<p class="muted">Median price per sq ft, for areas with at least 10 listings. '
                        "Most expensive first.</p>", unsafe_allow_html=True)
            medals = ["🥇", "🥈", "🥉"]
            rows = [((f"{medals[i]} " if i < 3 else "") + r.location, r.median_price_per_sqft)
                    for i, r in enumerate(s.head(12).itertuples())]
            st.markdown(ac.bars_html(rows, fmt=lambda v: f"Rs {v:,.0f}"), unsafe_allow_html=True)
            with st.expander(f"All {len(s)} areas in {pick}"):
                st.dataframe(s[["location", "listings", "median_price_per_sqft", "median_price_pkr"]].rename(columns={
                    "location": "Area", "listings": "Listings", "median_price_per_sqft": "Median per sq ft (Rs)",
                    "median_price_pkr": "Median price (Rs)"}), hide_index=True)

# ------------------------------------------------------------------ best value
if best_value is not None:
    with tabs["💎 Best-value listings"]:
        st.info("These listings are priced well below what similar listings suggest. Open the ad and check before "
                "acting: the model cannot see condition, floor level, lease title or unfinished construction, "
                "so some of these will have a good reason for the low price.")
        f1, f2 = st.columns(2)
        city_pick = f1.selectbox("City", ["All cities"] + sorted(best_value.city.unique()), key="bv_city")
        type_pick = f2.multiselect("Property type", sorted(best_value.property_type.unique()), key="bv_type")
        v = best_value
        if city_pick != "All cities":
            v = v[v.city == city_pick]
        if type_pick:
            v = v[v.property_type.isin(type_pick)]
        if v.empty:
            st.markdown('<p class="muted">No listings match. Clear a filter to see more.</p>', unsafe_allow_html=True)
        else:
            table = ac.format_listings(v.sort_values("price_ratio"))
            cfg = {"Ad": st.column_config.LinkColumn("Ad", display_text="Open ad")} if "Ad" in table else {}
            st.dataframe(table, hide_index=True, column_config=cfg)

# ------------------------------------------------------------------ about
with tabs["📊 About the numbers"]:
    st.markdown('<div class="h3">How accurate is it?</div>', unsafe_allow_html=True)
    st.markdown("Each model was tested on listings it never saw during training. "
                "Median error is how far off a typical estimate was, as a share of the asking price.")
    mt = ac.metrics_table(meta)
    if not mt.empty:
        st.dataframe(mt, hide_index=True)
    by_city = read_optional("reports/error_by_city.csv")
    if by_city is not None:
        by_city = by_city.rename(columns={"city": "City", "n": "Test listings", "R2": "R²", "MedianAPE_pct": "Median error (%)"})
        st.markdown("**By city** (small cities have few test listings, so their numbers move around more)")
        st.dataframe(by_city[["City", "Test listings", "R²", "Median error (%)"]].round(2), hide_index=True)

    rep = read_optional("reports/cleaning_report_2024.json", "json") or {}
    st.markdown('<div class="h3">What was fixed in the data</div>', unsafe_allow_html=True)
    fixes = []
    if rep.get("rows_raw"):
        fixes.append(f"Started with {rep['rows_raw']:,} listings and kept {rep.get('rows_clean', 0):,} houses, flats and "
                     "portions with a believable price and size.")
    if rep.get("dropped_content_duplicates"):
        fixes.append(f"Removed {rep['dropped_content_duplicates']:,} repeat ads for the same property, so the model "
                     "is never tested on a listing it has effectively seen.")
    if rep.get("dropped_installment_ads"):
        fixes.append(f"Removed {rep['dropped_installment_ads']:,} instalment and payment-plan ads, where the price shown "
                     "is a down payment, not the full price.")
    fixes.append("Converted every size to square feet from the ad text. The source file's own square-foot column was "
                 "wrong for square-yard listings in Karachi (it was 9 times too small).")
    fixes.append("Treated 0 bedrooms or bathrooms on a house as missing, not as a real zero.")
    st.markdown("\n".join(f"- {f}" for f in fixes))

    st.markdown('<div class="h3">What it cannot tell you</div>', unsafe_allow_html=True)
    st.markdown(
        "- **Asking prices, from one week in May 2024.** Final sale prices are usually lower, and prices have moved since.\n"
        "- **Condition, floor and finishing are invisible.** Grey-structure houses, leased titles and low-floor flats "
        "can sell far below the estimate.\n"
        "- **Prices vary a lot within the same area and size.** Two 1-kanal houses in the same society can differ by 2x.\n"
        "- **Small cities have fewer examples,** so estimates there are less reliable.\n"
        "- This is a statistical estimate, not a valuation or financial advice.")
