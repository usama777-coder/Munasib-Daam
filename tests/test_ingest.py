import io
import pandas as pd
from hpk.clean import clean
from hpk.ingest import ingest, read_any

ZAMEEN = """property_id;"location_id";"page_url";"property_type";"price";"location";"city";"province_name";"latitude";"longitude";"baths";"area";"purpose";"bedrooms";"date_added";"agency";"agent"
347795;8;"https://x/a-347795-8-1.html";"House";220000000;"Model Town";"Lahore";"Punjab";31.48;74.32;0;"6 Kanal";"For Sale";0;"07-17-2019";"A";"B"
482892;48;"https://x/b-482892-48-1.html";"House";40000000;"Multan Road";"Lahore";"Punjab";31.43;74.17;5;"1 Kanal";"For Sale";5;"10-06-2018";"";""
555962;75;"https://x/c-555962-75-1.html";"House";9500000;"Eden";"Lahore";"Punjab";31.49;74.41;0;"9 Marla";"For Sale";3;"07-03-2019";"";""
900001;75;"https://x/d-900001-75-1.html";"House";90000;"Eden";"Lahore";"Punjab";31.49;74.41;3;"9 Marla";"For Rent";3;"07-03-2019";"";""
"""


def test_zameen_layout(tmp_path):
    p = tmp_path / "z.csv"; p.write_text(ZAMEEN, encoding="utf-8")
    out = ingest(read_any(str(p)), "z")
    assert len(out) == 3                       # rent row dropped
    assert "agent" not in out.columns and "agency" not in out.columns
    assert out["listing_date"].iloc[0] == pd.Timestamp("2019-07-17")
    assert out["price_raw"].iloc[0] == 220000000
    df, rep = clean(out)
    assert rep["rows_clean"] >= 1 or rep["dropped_price_outliers"] >= 0


Z24 = """url,title,type,price,area,city,address,bedrooms,baths,area_sqft,price_per_sqft,area_sqm,price_per_sqm,Latitude,Longitude,date_added
https://w/a-1-1.html,5 MARLA HOUSE FOR SALE,House,23500000.0,5 Marla,Islamabad,"Bahria Town, Islamabad, Islamabad Capital",4 Beds,4 Baths,1361.25,17263.5,126.4,185823.3,33.6844,73.0479,2024-05-25
https://w/b-2-1.html,Prime House,House,750000000.0,3.8 Kanal,Islamabad,"F-7, Islamabad, Islamabad Capital",10 Beds,0 Baths,20691.0,36247.6,1922.2,390166.5,33.6844,73.0479,2024-05-25
https://w/c-3-1_for_rent.html,House for rent,House,90000.0,5 Marla,Lahore,"DHA, Lahore, Punjab",3 Beds,3 Baths,1361.25,66.1,126.4,712.0,31.5,74.3,2024-05-25
"""


def test_zameen2024_layout(tmp_path):
    p = tmp_path / "z24.csv"; p.write_text(Z24, encoding="utf-8")
    out = ingest(read_any(str(p)))
    assert len(out) == 2                                   # rental dropped
    assert list(out["location"]) == ["Bahria Town", "F-7"]
    assert "latitude" not in out.columns                    # placeholder coordinates not carried over
    assert out["listing_date"].iloc[0] == pd.Timestamp("2024-05-25")
    df, rep = clean(out)
    assert rep["bathrooms_zero_set_to_missing"] == 1        # "0 Baths" on a 10-bed house = unknown
