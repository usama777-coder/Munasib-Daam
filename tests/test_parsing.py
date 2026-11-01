import pytest
from hpk.parsing import MARLA_SQFT, parse_area_sqft, parse_count, parse_price


@pytest.mark.parametrize("text,expected", [
    ("1.5 Crore", 15_000_000),
    ("85 Lakh", 8_500_000),
    ("85 Lac", 8_500_000),
    ("1 Crore 20 Lakh", 12_000_000),
    ("2.1 Arab", 2_100_000_000),
    ("PKR 4,500,000", 4_500_000),
    ("Rs. 45 Lakh", 4_500_000),
    ("45 lakh", 4_500_000),
    (7_500_000, 7_500_000),
    ("Call for price", None),
    ("", None),
    (None, None),
    ("abc", None),
])
def test_parse_price(text, expected):
    assert parse_price(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("5 Marla", 5 * MARLA_SQFT),
    ("1 Kanal", 20 * MARLA_SQFT),
    ("1 Kanal 5 Marla", 25 * MARLA_SQFT),
    ("10 Marla 9 Sarsai", 11 * MARLA_SQFT),
    ("200 Sq. Yd.", 1800),
    ("120 Square Yards", 1080),
    ("1,800 sqft", 1800),
    ("1800 Square Feet", 1800),
    ("100 Square Meters", pytest.approx(1076.39)),
    ("2 Acre", 87_120),
    ("nonsense", None),
    (None, None),
])
def test_parse_area(text, expected):
    assert parse_area_sqft(text) == expected


@pytest.mark.parametrize("text,expected", [("5", 5), ("4 Beds", 4), ("Studio", 0), (None, None), ("-", None)])
def test_parse_count(text, expected):
    assert parse_count(text) == expected
