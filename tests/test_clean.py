import pandas as pd
from hpk.clean import clean
from hpk.synth import make_synthetic


def test_clean_synthetic():
    raw = make_synthetic(1500, seed=1)
    df, rep = clean(raw)
    assert rep["rows_clean"] == len(df) > 1000
    assert df["price_pkr"].gt(0).all() and df["area_sqft"].gt(0).all()
    assert not df[["bedrooms", "bathrooms"]].isna().any().any()
    assert set(df["city"]) <= {"Islamabad", "Lahore", "Karachi", "Peshawar"}
    assert rep["dropped_unparseable_price"] > 0  # "Call for price" rows removed
    assert rep["dropped_duplicates"] > 0


def test_unknown_city_dropped():
    raw = make_synthetic(300, seed=2)
    raw.loc[0, "city"] = "Atlantis"
    _, rep = clean(raw)
    assert rep["dropped_unknown_city"] == 1
