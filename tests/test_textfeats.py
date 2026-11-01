import pandas as pd
import pytest
from hpk.modeling import add_features
from hpk.textfeats import INSTALLMENT, is_plot_ad, slug_text, subarea
import re

U = "https://www.zameen.com/Property/{}.html"


@pytest.mark.parametrize("slug,expected", [
    ("dha_defence_dha_phase_1_flat_for_sale-49147525-1478-1", "phase 1"),
    ("dha_phase_5_badar_commercial_area_apartment_for_sale_in_badar_commercial_phase_5-49787055-8484-1", "phase 5"),
    ("park_view_city_park_view_-_block_j_house_for_sale_in_j_block_park_view_city-49564948-16867-1", "block j"),
    ("islamabad_f-7_prime_location_house_for_sale-47991532-165-1", "none"),
    ("gulberg_greens_sector_c_house-1234-5-1", "sector c"),
])
def test_subarea(slug, expected):
    assert subarea(slug_text(U.format(slug))) == expected


@pytest.mark.parametrize("slug,is_inst", [
    ("dha_defence_dha_phase_6_apartment_for_sale_25_down_payment-49779541-1448-1", True),
    ("dha_phase_2_dha_phase_2_-_block_q_3_bed_apartment_for_sale_on_installment_plan-49570012-1570-1", True),
    ("bahria_town_bahria_enclave_5_marla_brand_new_park_facing_house_for_sale-49690085-1705-1", False),
])
def test_installment(slug, is_inst):
    assert bool(re.search(INSTALLMENT, slug_text(U.format(slug)))) is is_inst


def test_flags_and_overrides():
    url = U.format("bahria_town_bahria_enclave_5_marla_brand_new_park_facing_house_for_sale-49690085-1705-1")
    df = pd.DataFrame({"city": ["Islamabad"], "location": ["Bahria Town"], "url": [url], "area_sqft": [1361.25]})
    out = add_features(df)
    assert out.loc[0, "brand_new"] == 1 and out.loc[0, "park_facing"] == 1 and out.loc[0, "corner"] == 0
    # "Park View City" must NOT count as park facing
    pv = pd.DataFrame({"city": ["Islamabad"], "location": ["Park View City"], "area_sqft": [3000.0],
                       "url": [U.format("park_view_city_park_view_-_block_j_house-1-2-3")]})
    assert add_features(pv).loc[0, "park_facing"] == 0
    # caller-supplied values (the app) are respected, and no url is fine
    app = pd.DataFrame({"city": ["Lahore"], "location": ["Dha"], "area_sqft": [2000.0], "subarea": ["phase 5"], "corner": [1]})
    o = add_features(app)
    assert o.loc[0, "area_sub"] == "Lahore | Dha | phase 5" and o.loc[0, "corner"] == 1 and o.loc[0, "basement"] == 0


@pytest.mark.parametrize("slug,expected", [
    ("dha_phase_4_goldcrest_mall_residency_1_bed_available_for_sale_pay_in_3_years_get_possession_in_1_5_years-1-2-3", True),
    ("house_for_sale_ready_possession_immediate-1-2-3", False),
    ("5_marla_house_for_sale_with_immediate_possession-1-2-3", False),
])
def test_payment_plan_phrases(slug, expected):
    assert bool(re.search(INSTALLMENT, slug_text(U.format(slug)))) is expected


@pytest.mark.parametrize("slug,expected", [
    ("scheme_33_saadi_garden_prime_living_awaits_west_open_120_sq_yd_plot_in_vip_block_2-1-2-3", True),
    ("house_built_on_10_marla_plot_in_bahria-1-2-3", False),
    ("bin_qasim_town_shah_latif_town_120_sq_yard_house_for_sale-1-2-3", False),
])
def test_plot_ads(slug, expected):
    assert is_plot_ad(slug_text(U.format(slug))) is expected


def test_leased_flag():
    df = pd.DataFrame({"city": ["Karachi"], "location": ["Gadap Road"], "area_sqft": [1647.0],
                       "url": [U.format("gadap_road_commander_enclave_leased_apartment_for_sale-1-2-3")]})
    assert add_features(df).loc[0, "leased"] == 1
