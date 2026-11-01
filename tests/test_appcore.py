import numpy as np
import pandas as pd
import pytest
from hpk import appcore as ac


def test_pkr():
    assert ac.pkr(25_900_000) == "Rs 2.59 crore"
    assert ac.pkr(8_500_000) == "Rs 85.0 lakh"
    assert ac.pkr(45_000) == "Rs 45,000"


@pytest.mark.parametrize("asking,tone", [(5, "good"), (7.5, "good"), (10, "calm"), (11.4, "calm"), (13, "warn"), (20, "bad")])
def test_deal_verdict_bands(asking, tone):
    assert ac.deal_verdict(asking, 10.0)["tone"] == tone


def test_driver_rows_sorted_labelled_and_filtered():
    c = pd.Series({"area": 0.20, "log_area": -0.10, "bedrooms": 0.001, "brand_new": 0.05, "city": 0.0})
    rows = ac.driver_rows(c)
    assert [r[0] for r in rows] == ["Area or society", "Size", "Brand new"]
    assert rows[0][1] == pytest.approx((np.exp(0.2) - 1) * 100)


def test_area_helpers():
    stats = pd.DataFrame({"city": ["A", "A", "B"], "location": ["x", "y", "z"], "listings": [10, 90, 5],
                          "median_price_per_sqft": [100.0, 200.0, 50.0]})
    assert ac.area_context(stats, "A", "y") == {"listings": 90, "median_ppsf": 200.0}
    assert ac.area_context(stats, "A", "nope") is None
    assert ac.popular_area(stats, {"A": ["x", "y"], "B": ["z"]}, "A") == "y"
    assert ac.popular_area(pd.DataFrame(), {"A": ["x", "y"]}, "A") == "x"
    cs = ac.city_summary(stats)
    assert list(cs.city) == ["A", "B"] and cs.loc[0, "typical_ppsf"] == pytest.approx(190.0)


def test_ruler_marks_estimate_and_asking_inside_track():
    h = ac.ruler_html(2.6e7, 2.0e7, 3.3e7, asking=4.0e7)
    assert "Estimate" in h and "Asking Rs 4.00 crore" in h and "rband" in h
    for part in h.split("left:")[1:]:
        assert 0 <= float(part.split("%")[0]) <= 100
    assert "Asking" not in ac.ruler_html(2.6e7, 2.0e7, 3.3e7)


def test_bars_html_signed_unsigned_and_escaping():
    s = ac.bars_html([("Area <b>", 12.0), ("Size", -8.0)], signed=True)
    assert "Area &lt;b&gt;" in s and "<b>" not in s and "bfill pos" in s and "bfill neg" in s and "+12%" in s
    u = ac.bars_html([("A", 200.0), ("B", 100.0)])
    assert "width:100.0%" in u and "width:50.0%" in u
    assert "nothing" in ac.bars_html([], empty="nothing")


def test_format_listings():
    d = pd.DataFrame({"city": ["Lahore"], "location": ["Dha"], "property_type": ["House"], "area_sqft": [2722.5],
                      "bedrooms": [4], "price_pkr": [1.5e7], "fair_price_pkr": [3.0e7], "price_ratio": [0.5], "url": ["u"]})
    f = ac.format_listings(d)
    assert f.loc[0, "Below estimate (%)"] == 50 and f.loc[0, "Listed price"] == "Rs 1.50 crore" and f.loc[0, "Ad"] == "u"


def test_metrics_table():
    meta = {"metrics": {"A": {"R2": 0.9, "MedianAPE_pct": 14.0, "MAE_pkr": 1e6}, "B": {"R2": 0.8, "MedianAPE_pct": 18.0, "MAE_pkr": 2e6}}}
    t = ac.metrics_table(meta)
    assert list(t.Model) == ["A", "B"] and t.loc[t.Model == "A", "Average error (Rs lakh)"].iloc[0] == 10.0


def test_theme_css_and_default_fallback():
    css = ac.theme_css("Indus")
    assert "--primary:#3b63c9" in css and "--hero-a:" in css and css.startswith("<style>")
    assert ac.theme_css("nonsense") == ac.theme_css(ac.DEFAULT_THEME)
    for name, t in ac.THEMES.items():
        assert set(t) == set(ac.THEMES[ac.DEFAULT_THEME]), name


@pytest.mark.parametrize("x,expected", [(25_900_000, "2 crore 59 lakh"), (8_500_000, "85 lakh"),
                                        (20_000_000, "2 crore"), (99_960_000, "10 crore")])
def test_pkr_words(x, expected):
    assert ac.pkr_words(x) == expected


def test_title_band_is_separate_from_intro_and_tiles_escape():
    h = ac.hero_html()
    assert ac.APP_NAME == "Munasib Daam" and ac.APP_NAME in h and 'class="stripe"' in h
    assert ac.TAGLINE not in h and "37,042" not in h and "tagline" not in h, "tagline/description must not be in the title band"
    i = ac.intro_html(37042, 9, 13.7)
    assert ac.TAGLINE in i and "37,042" in i and "9 Pakistani cities" in i and "14%" in i and ac.APP_NAME not in i
    assert "hero" not in i
    t = ac.tiles_html([ac.tile_html("A <i>", "Rs 1", "sub", "s2"), ac.tile_html("B", "Rs 2")])
    assert "A &lt;i&gt;" in t and "<i>" not in t and "var(--s2)" in t and t.count('class="tile"') == 2


def test_every_theme_has_valid_page_background_colours():
    import re
    for name, th in ac.THEMES.items():
        for k in ("bg_a", "bg_b", "bg_c", "bg_d", "primary", "hero_a", "hero_b"):
            assert re.fullmatch(r"#[0-9a-f]{6}", th[k]), (name, k)
    assert "--bg-a:" in ac.theme_css("Sabz") and "--bg-d:" in ac.theme_css("Sunset")


def test_driver_rows_labels_include_the_users_values():
    c = pd.Series({"log_area": -0.4, "city": -0.1, "brand_new": 0.05})
    rows = ac.driver_rows(c, details={"log_area": "5 marla", "city": "Peshawar", "area_sub": None})
    labels = [r[0] for r in rows]
    assert "Size (5 marla)" in labels and "City (Peshawar)" in labels and "Brand new" in labels
    assert rows[0][1] < 0   # smaller than the average listing -> negative, by design
