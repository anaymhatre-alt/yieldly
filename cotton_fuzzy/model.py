"""Hierarchical zero-order Sugeno fuzzy model for cotton yield.

Level 1 (two modules)
    water   : season water ratio x critical-stage water ratio  -> water index
    thermal : mean temperature  x hot days in boll set         -> thermal index
Level 2
    final   : water index x thermal index x soil water capacity -> yield fraction

Each rule's output is a number between 0 and 1 (zero-order Sugeno). The result
is the firing-strength-weighted average of the rule outputs, multiplied by the
potential yield. Rule outputs come from an expert prior (priors.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import priors
from .data import DEFAULTS, prepare_frame
from .membership import partition_memberships

SEASON_CENTERS = (0.0, 0.4, 0.75, 1.0, 1.5)
SEASON_LABELS = ("none", "low", "moderate", "adequate", "excess")
CRITICAL_CENTERS = SEASON_CENTERS
CRITICAL_LABELS = ("none", "low", "moderate", "adequate", "ample")
TEMP_CENTERS = (16.0, 22.0, 27.0, 31.0, 36.0)
TEMP_LABELS = ("cold", "cool", "optimal", "warm", "hot")
HOT_CENTERS = (0.0, 4.0, 10.0, 20.0)
HOT_LABELS = ("none", "few", "many", "severe")
INDEX_CENTERS = (0.2, 0.6, 1.0)
INDEX_LABELS = ("poor", "fair", "good")
SOIL_CENTERS = (60.0, 120.0, 180.0)
SOIL_LABELS = ("light", "medium", "deep")

ADJUSTABLE_FIELDS = ("rainfall_mm", "irrigation_mm", "mean_temp_c", "hot_days", "soil_awc_mm")


def _prior_arrays():
    return (
        priors.water_rules(SEASON_CENTERS, CRITICAL_CENTERS),
        priors.thermal_rules(TEMP_CENTERS, HOT_CENTERS),
        priors.final_rules(INDEX_CENTERS, INDEX_CENTERS, SOIL_CENTERS),
    )


@dataclass
class ModelParams:
    potential_yield_t_ha: float = 2.5
    water_requirement_mm: float = 900.0
    rainfall_effectiveness: float = 0.8
    critical_share: float = 0.5
    water_rules: Optional[np.ndarray] = None
    thermal_rules: Optional[np.ndarray] = None
    final_rules: Optional[np.ndarray] = None

    def __post_init__(self):
        w, t, f = _prior_arrays()
        self.water_rules = w if self.water_rules is None else np.array(self.water_rules, dtype=float)
        self.thermal_rules = t if self.thermal_rules is None else np.array(self.thermal_rules, dtype=float)
        self.final_rules = f if self.final_rules is None else np.array(self.final_rules, dtype=float)
        if self.water_rules.shape != w.shape or self.thermal_rules.shape != t.shape or self.final_rules.shape != f.shape:
            raise ValueError("Rule tables have the wrong shape for this model version.")
        if self.potential_yield_t_ha <= 0 or self.water_requirement_mm <= 0:
            raise ValueError("Potential yield and water requirement must be positive.")
        if not 0 < self.rainfall_effectiveness <= 1.0 or not 0 < self.critical_share <= 1.0:
            raise ValueError("Rainfall effectiveness and critical share must be in (0, 1].")

    def copy_with(self, **changes) -> "ModelParams":
        return replace(self, **changes)


@dataclass
class Scenario:
    area_ha: float = 1.0
    rainfall_mm: float = 650.0
    irrigation_mm: float = 200.0
    irrigation_efficiency: float = 0.7
    critical_rainfall_mm: Optional[float] = None
    critical_irrigation_mm: Optional[float] = None
    mean_temp_c: float = DEFAULTS["mean_temp_c"]
    hot_days: float = DEFAULTS["hot_days"]
    soil_awc_mm: float = DEFAULTS["soil_awc_mm"]

    def to_frame(self) -> pd.DataFrame:
        row = {
            "rainfall_mm": self.rainfall_mm,
            "irrigation_mm": self.irrigation_mm,
            "irrigation_efficiency": self.irrigation_efficiency,
            "critical_rainfall_mm": np.nan if self.critical_rainfall_mm is None else self.critical_rainfall_mm,
            "critical_irrigation_mm": np.nan if self.critical_irrigation_mm is None else self.critical_irrigation_mm,
            "mean_temp_c": self.mean_temp_c,
            "hot_days": self.hot_days,
            "soil_awc_mm": self.soil_awc_mm,
        }
        return pd.DataFrame([row])


@dataclass
class Inputs:
    """Numpy view of a validated table, so the model can be evaluated fast."""

    rain: np.ndarray
    irr: np.ndarray
    eta: np.ndarray
    crit_rain: np.ndarray
    crit_irr: np.ndarray
    temp: np.ndarray
    hot: np.ndarray
    awc: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Inputs":
        d, _ = prepare_frame(df)
        g = lambda c: d[c].to_numpy(dtype=float)
        return cls(g("rainfall_mm"), g("irrigation_mm"), g("irrigation_efficiency"), g("critical_rainfall_mm"),
                   g("critical_irrigation_mm"), g("mean_temp_c"), g("hot_days"), g("soil_awc_mm"))

    def __len__(self) -> int:
        return len(self.rain)


def _sugeno(memberships: List[np.ndarray], rules: np.ndarray) -> np.ndarray:
    letters = "ijk"[: len(memberships)]
    weights = np.einsum(",".join("n" + c for c in letters) + "->n" + letters, *memberships)
    num = np.einsum("n" + letters + "," + letters + "->n", weights, rules)
    den = weights.reshape(len(weights), -1).sum(axis=1)
    return num / den


class CottonFuzzyModel:
    def __init__(self, params: Optional[ModelParams] = None):
        self.params = params or ModelParams()

    # ---- core evaluation -------------------------------------------------
    def _ratios(self, x: Inputs):
        p = self.params
        swr = (p.rainfall_effectiveness * x.rain + x.eta * x.irr) / p.water_requirement_mm
        has_crit = ~(np.isnan(x.crit_rain) & np.isnan(x.crit_irr))
        crit_eff = p.rainfall_effectiveness * np.nan_to_num(x.crit_rain) + x.eta * np.nan_to_num(x.crit_irr)
        cwr = np.where(has_crit, crit_eff / (p.critical_share * p.water_requirement_mm), swr)
        return swr, cwr

    def _final(self, w, t, awc):
        mu = [partition_memberships(w, INDEX_CENTERS), partition_memberships(t, INDEX_CENTERS),
              partition_memberships(awc, SOIL_CENTERS)]
        return np.clip(_sugeno(mu, self.params.final_rules), 0.0, 1.0), mu

    def forward(self, x: Inputs) -> Dict[str, object]:
        swr, cwr = self._ratios(x)
        mu_season = partition_memberships(swr, SEASON_CENTERS)
        mu_crit = partition_memberships(cwr, CRITICAL_CENTERS)
        mu_temp = partition_memberships(x.temp, TEMP_CENTERS)
        mu_hot = partition_memberships(x.hot, HOT_CENTERS)
        water_index = _sugeno([mu_season, mu_crit], self.params.water_rules)
        thermal_index = _sugeno([mu_temp, mu_hot], self.params.thermal_rules)
        fraction, mu_final = self._final(water_index, thermal_index, x.awc)
        return {
            "fraction": fraction, "water_index": water_index, "thermal_index": thermal_index,
            "season_ratio": swr, "critical_ratio": cwr,
            "mu_season": mu_season, "mu_crit": mu_crit, "mu_temp": mu_temp, "mu_hot": mu_hot, "mu_final": mu_final,
        }

    def predict_inputs(self, x: Inputs) -> np.ndarray:
        """Yield in t/ha for each row."""
        return self.params.potential_yield_t_ha * self.forward(x)["fraction"]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_inputs(Inputs.from_frame(df))

    # ---- single scenario with explanation --------------------------------
    def estimate(self, s: Scenario) -> dict:
        if s.area_ha <= 0:
            raise ValueError("Area must be greater than zero.")
        x = Inputs.from_frame(s.to_frame())
        out = self.forward(x)
        p = self.params
        frac = float(out["fraction"][0])
        per_ha = p.potential_yield_t_ha * frac

        w, t, awc = out["water_index"], out["thermal_index"], x.awc
        headroom = {
            "Water supply": float(self._final(np.ones(1), t, awc)[0][0] - frac),
            "Temperature and heat": float(self._final(w, np.ones(1), awc)[0][0] - frac),
            "Soil water capacity": float(self._final(w, t, np.full(1, SOIL_CENTERS[-1]))[0][0] - frac),
        }
        headroom = {k: max(0.0, v) * p.potential_yield_t_ha for k, v in headroom.items()}

        def labelled(mu, labels):
            return {lab: float(mu[0, i]) for i, lab in enumerate(labels)}

        return {
            "yield_fraction": frac,
            "yield_per_ha_t": per_ha,
            "total_yield_t": per_ha * s.area_ha,
            "water_index": float(w[0]),
            "thermal_index": float(t[0]),
            "season_water_ratio": float(out["season_ratio"][0]),
            "critical_water_ratio": float(out["critical_ratio"][0]),
            "headroom_t_ha": headroom,
            "memberships": {
                "Season water": labelled(out["mu_season"], SEASON_LABELS),
                "Critical-stage water": labelled(out["mu_crit"], CRITICAL_LABELS),
                "Mean temperature": labelled(out["mu_temp"], TEMP_LABELS),
                "Hot days": labelled(out["mu_hot"], HOT_LABELS),
                "Soil capacity": labelled(out["mu_final"][2], SOIL_LABELS),
            },
            "rules": self._active_rules(out),
        }

    def _active_rules(self, out, min_strength: float = 0.01) -> pd.DataFrame:
        p = self.params
        rows = []
        def add(module, mus, labels, rules, template):
            grids = np.einsum(",".join("n" + c for c in "ijk"[: len(mus)]) + "->n" + "ijk"[: len(mus)], *mus)[0]
            for idx in np.ndindex(grids.shape):
                if grids[idx] >= min_strength:
                    names = [labels[d][i] for d, i in enumerate(idx)]
                    rows.append({"module": module, "rule": template.format(*names),
                                 "strength": float(grids[idx]), "output": float(rules[idx])})
        add("Water", [out["mu_season"], out["mu_crit"]], [SEASON_LABELS, CRITICAL_LABELS], p.water_rules,
            "IF season water is {} AND critical-stage water is {} THEN water index")
        add("Thermal", [out["mu_temp"], out["mu_hot"]], [TEMP_LABELS, HOT_LABELS], p.thermal_rules,
            "IF temperature is {} AND hot days are {} THEN thermal index")
        add("Final", out["mu_final"], [INDEX_LABELS, INDEX_LABELS, SOIL_LABELS], p.final_rules,
            "IF water index is {} AND thermal index is {} AND soil is {} THEN yield fraction")
        df = pd.DataFrame(rows, columns=["module", "rule", "strength", "output"])
        return df.sort_values(["module", "strength"], ascending=[True, False]).reset_index(drop=True)

    # ---- what-if helpers -------------------------------------------------
    def response_curve(self, s: Scenario, field: str, values) -> np.ndarray:
        """Yield (t/ha) as one input varies and the rest of the scenario is held fixed."""
        if field not in ADJUSTABLE_FIELDS:
            raise ValueError(f"field must be one of {ADJUSTABLE_FIELDS}")
        values = np.asarray(values, dtype=float)
        df = pd.concat([s.to_frame()] * len(values), ignore_index=True)
        df[field] = values
        base = getattr(s, field)
        crit_col = {"rainfall_mm": "critical_rainfall_mm", "irrigation_mm": "critical_irrigation_mm"}.get(field)
        if crit_col and base and not df[crit_col].isna().all():
            df[crit_col] = df[crit_col] * values / base  # keep the critical-stage share constant
        return self.predict(df)
