"""Mamdani fuzzy controller for cotton yield estimation."""
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CottonProfile:
    potential_yield_t_ha: float = 2.5
    optimal_rainfall_mm: float = 700
    adequate_irrigation_mm: float = 300


def triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function, including shoulder endpoints."""
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return 1.0 if a == b else (x - a) / (b - a)
    return 1.0 if b == c else (c - x) / (c - b)


def memberships(value: float, sets: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
    return {name: triangular(value, *bounds) for name, bounds in sets.items()}


def estimate_cotton_yield(area_ha: float, rainfall_mm: float, irrigation_mm: float, profile: CottonProfile) -> dict:
    """Estimate cotton yield with Mamdani rules for rainfall and irrigation.

    Rainfall and irrigation fuzzy sets scale with the configured crop profile.
    Output is a fraction of the configured potential yield.
    """
    if area_ha <= 0:
        raise ValueError("Area must be greater than zero.")
    if rainfall_mm < 0 or irrigation_mm < 0:
        raise ValueError("Rainfall and irrigation cannot be negative.")
    if profile.potential_yield_t_ha <= 0 or profile.optimal_rainfall_mm <= 0 or profile.adequate_irrigation_mm <= 0:
        raise ValueError("Potential yield and water reference values must be positive.")

    optimum = profile.optimal_rainfall_mm
    rain = memberships(rainfall_mm, {
        "low": (0, 0, 0.8 * optimum),
        "normal": (0.4 * optimum, optimum, 1.6 * optimum),
        "high": (1.2 * optimum, 2.0 * optimum, 2.0 * optimum),
    })
    irrigation_ref = profile.adequate_irrigation_mm
    irrigation = memberships(irrigation_mm, {
        "low": (0, 0, 0.8 * irrigation_ref),
        "medium": (0.4 * irrigation_ref, irrigation_ref, 1.6 * irrigation_ref),
        "high": (1.2 * irrigation_ref, 2.0 * irrigation_ref, 2.0 * irrigation_ref),
    })

    yield_sets = {
        "poor": (0.0, 0.0, 0.40),
        "moderate": (0.20, 0.50, 0.80),
        "good": (0.60, 0.82, 1.0),
        "excellent": (0.82, 1.0, 1.0),
    }
    # IF rainfall is ... AND irrigation is ... THEN yield is ...
    rules = {
        ("low", "low"): "poor",
        ("low", "medium"): "moderate",
        ("low", "high"): "good",
        ("normal", "low"): "moderate",
        ("normal", "medium"): "good",
        ("normal", "high"): "excellent",
        ("high", "low"): "moderate",
        ("high", "medium"): "moderate",
        ("high", "high"): "poor",
    }
    resolution = 1001
    xs = [i / (resolution - 1) for i in range(resolution)]
    aggregate = [0.0] * resolution
    for (rain_level, irrigation_level), quality in rules.items():
        strength = min(rain[rain_level], irrigation[irrigation_level])
        if strength <= 0:
            continue
        a, b, c = yield_sets[quality]
        for i, x in enumerate(xs):
            aggregate[i] = max(aggregate[i], min(strength, triangular(x, a, b, c)))

    denominator = sum(aggregate)
    fraction = 0.20 if denominator == 0 else sum(x * mu for x, mu in zip(xs, aggregate)) / denominator
    yield_per_ha = fraction * profile.potential_yield_t_ha
    return {
        "yield_fraction": fraction,
        "yield_per_ha_t": yield_per_ha,
        "total_yield_t": yield_per_ha * area_ha,
        "rainfall_memberships": rain,
        "irrigation_memberships": irrigation,
    }
