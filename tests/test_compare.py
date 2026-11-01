import pandas as pd
from hpk.compare_years import growth


def test_growth():
    old = pd.DataFrame({"city": ["A"] * 40, "price_per_sqft": [1000.0] * 40})
    new = pd.DataFrame({"city": ["A"] * 40, "price_per_sqft": [2500.0] * 40})
    g = growth(old, new, ["city"])
    assert round(g.growth_pct.iloc[0]) == 150
