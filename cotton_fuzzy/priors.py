"""Expert prior for the rule consequents.

These numbers are starting points, not facts. The water part follows the shape
of the FAO-33 yield-response idea (relative yield falls roughly linearly with
relative water deficit, total Ky about 0.85 for cotton) and splits the loss
between the whole season and the flowering/boll period. The temperature part is
a smooth penalty away from a mild optimum plus a penalty for very hot days.
They are illustrative and not validated for any region.
"""
from __future__ import annotations

import numpy as np

KY_SEASON = 0.40
KY_CRITICAL = 0.50
EXCESS_LOSS_AT_MAX = 0.30  # relative yield loss when season water reaches the top set centre


def water_rules(season_centers, critical_centers) -> np.ndarray:
    s = np.asarray(season_centers, dtype=float)[:, None]
    c = np.asarray(critical_centers, dtype=float)[None, :]
    deficit_season = np.maximum(0.0, 1.0 - s)
    deficit_critical = np.maximum(0.0, 1.0 - c)
    span = max(float(s.max()) - 1.0, 1e-9)
    excess = 1.0 - EXCESS_LOSS_AT_MAX * np.clip((s - 1.0) / span, 0.0, 1.0)
    return (1.0 - KY_SEASON * deficit_season) * (1.0 - KY_CRITICAL * deficit_critical) * excess


def thermal_rules(temp_centers, hot_centers) -> np.ndarray:
    t = np.asarray(temp_centers, dtype=float)[:, None]
    h = np.asarray(hot_centers, dtype=float)[None, :]
    deviation = np.maximum(0.0, np.abs(t - 27.0) - 2.5)
    temp_factor = 1.0 - np.minimum(0.7, 0.012 * deviation**2)
    hot_factor = 1.0 - np.minimum(0.6, 0.02 * h)
    return temp_factor * hot_factor


def final_rules(water_centers, thermal_centers, soil_centers) -> np.ndarray:
    w = np.asarray(water_centers, dtype=float)[:, None, None]
    t = np.asarray(thermal_centers, dtype=float)[None, :, None]
    a = np.asarray(soil_centers, dtype=float)[None, None, :]
    soil_norm = np.clip((a - 60.0) / 120.0, 0.0, 1.0)
    # A deeper soil buffers dry spells, so it matters only when water is limiting.
    soil_modifier = 1.0 + 0.10 * (soil_norm - 0.5) * (1.0 - w)
    return np.clip(w * t * soil_modifier, 0.02, 1.0)
