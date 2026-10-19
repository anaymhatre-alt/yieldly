import numpy as np
import pandas as pd
import pytest

from cotton_fuzzy import CottonFuzzyModel, Scenario
from cotton_fuzzy import data as D
from cotton_fuzzy.membership import partition_memberships


def test_partition_sums_to_one_and_covers_everything():
    centers = (0.0, 0.4, 0.75, 1.0, 1.5)
    x = np.concatenate([np.linspace(-5, 10, 2001), [0, 0.4, 0.75, 1.0, 1.5]])
    mu = partition_memberships(x, centers)
    assert np.allclose(mu.sum(axis=1), 1.0)
    assert (mu >= 0).all() and (mu <= 1).all()


def test_partition_rejects_bad_input():
    with pytest.raises(ValueError):
        partition_memberships([1.0], (1.0, 1.0, 2.0))
    with pytest.raises(ValueError):
        partition_memberships([np.nan], (0.0, 1.0))


def test_extreme_inputs_have_no_fallback_or_nan():
    m = CottonFuzzyModel()
    for rain in (0, 700, 5000):
        for irr in (0, 300, 4000):
            e = m.estimate(Scenario(rainfall_mm=rain, irrigation_mm=irr))
            assert 0.0 <= e["yield_fraction"] <= 1.0
            assert np.isfinite(e["yield_per_ha_t"])


def test_no_jumps_across_input_range():
    """The old controller dropped to a fixed 0.20 beyond 2x the reference. Check the curve is continuous."""
    m = CottonFuzzyModel()
    s = Scenario(mean_temp_c=27, hot_days=0)
    for field, hi in (("rainfall_mm", 3000), ("irrigation_mm", 3000)):
        xs = np.linspace(0, hi, 3001)
        y = m.response_curve(s, field, xs)
        assert np.max(np.abs(np.diff(y))) < 0.01


def test_more_water_never_hurts_until_excess():
    m = CottonFuzzyModel()
    s = Scenario(rainfall_mm=0, mean_temp_c=27, hot_days=0)
    xs = np.linspace(0, 700, 100)  # season ratio stays at or below 1 for these values
    y = m.response_curve(s, "irrigation_mm", xs)
    assert np.all(np.diff(y) >= -1e-9)


def test_rainfall_has_no_dead_zone_around_optimum():
    m = CottonFuzzyModel()
    s = Scenario(irrigation_mm=0, mean_temp_c=27, hot_days=0)
    y = m.response_curve(s, "rainfall_mm", [600, 700, 800])
    assert y[0] < y[1] < y[2]


def test_heat_lowers_yield_and_ideal_case_reaches_potential():
    m = CottonFuzzyModel()
    # 1125 mm of rain x 0.8 effectiveness = 900 mm = exactly the default water requirement
    base = Scenario(rainfall_mm=1125, irrigation_mm=0, mean_temp_c=27, hot_days=0)
    ideal = m.estimate(base)
    assert ideal["yield_fraction"] == pytest.approx(1.0, abs=0.02)
    hot = m.estimate(Scenario(**{**base.__dict__, "hot_days": 25}))
    assert hot["yield_fraction"] < ideal["yield_fraction"] - 0.1


def test_critical_stage_water_matters():
    m = CottonFuzzyModel()
    even = Scenario(rainfall_mm=600, irrigation_mm=0)
    dry_flowering = Scenario(rainfall_mm=600, irrigation_mm=0, critical_rainfall_mm=60, critical_irrigation_mm=0)
    assert m.estimate(dry_flowering)["yield_per_ha_t"] < m.estimate(even)["yield_per_ha_t"]


def test_area_scales_total_only():
    m = CottonFuzzyModel()
    a = m.estimate(Scenario(area_ha=1))
    b = m.estimate(Scenario(area_ha=12))
    assert b["yield_per_ha_t"] == pytest.approx(a["yield_per_ha_t"])
    assert b["total_yield_t"] == pytest.approx(12 * a["total_yield_t"])
    with pytest.raises(ValueError):
        m.estimate(Scenario(area_ha=0))


def test_vectorised_matches_single_estimate():
    m = CottonFuzzyModel()
    s = Scenario(rainfall_mm=480, irrigation_mm=120, mean_temp_c=30, hot_days=12, soil_awc_mm=150)
    assert m.predict(s.to_frame())[0] == pytest.approx(m.estimate(s)["yield_per_ha_t"])


def test_input_validation_defaults_and_errors():
    df = pd.DataFrame({"rainfall_mm": [500, 600], "irrigation_mm": [0, 100]})
    out, warns = D.prepare_frame(df)
    assert out["mean_temp_c"].eq(D.DEFAULTS["mean_temp_c"]).all()
    assert any("mean_temp_c" in w for w in warns)
    with pytest.raises(ValueError, match="Missing required"):
        D.prepare_frame(df.drop(columns=["rainfall_mm"]))
    with pytest.raises(ValueError, match="non-numeric"):
        D.prepare_frame(df.assign(rainfall_mm=["a", "b"]))
    with pytest.raises(ValueError, match="plausible range"):
        D.prepare_frame(df.assign(rainfall_mm=[500, 99999]))
