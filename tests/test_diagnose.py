import pandas as pd
from hpk.clean import clean
from hpk.diagnose_area import diagnose
from hpk.synth import make_synthetic


def test_diagnose_flags_disagreement():
    df = pd.DataFrame({"area": ["5 Marla", "5 Marla"], "area_sqft": [1361.25, 1125.0], "city": ["a", "a"], "type": ["House"] * 2})
    d = diagnose(df)
    assert list(d.agree) == [True, False]


def test_property_type_filter():
    raw = make_synthetic(500, seed=7)
    raw.loc[:49, "property_type"] = "Residential Plot"
    df, rep = clean(raw, property_types=["House", "Flat", "Upper Portion"])
    assert "Residential Plot" not in set(df["property_type"])
    assert rep["dropped_other_property_types"] >= 50
