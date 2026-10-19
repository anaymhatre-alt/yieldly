"""Input table validation: required columns, numeric types, plausible ranges and documented defaults."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

TARGET = "yield_t_ha"
REQUIRED = ["rainfall_mm", "irrigation_mm"]
DEFAULTS = {
    "irrigation_efficiency": 0.7,
    "mean_temp_c": 28.0,
    "hot_days": 5.0,
    "soil_awc_mm": 120.0,
}
OPTIONAL_NAN = ["critical_rainfall_mm", "critical_irrigation_mm"]
FEATURE_COLUMNS = REQUIRED + list(DEFAULTS) + OPTIONAL_NAN
RANGES = {
    "rainfall_mm": (0.0, 5000.0),
    "irrigation_mm": (0.0, 4000.0),
    "irrigation_efficiency": (0.1, 1.0),
    "critical_rainfall_mm": (0.0, 5000.0),
    "critical_irrigation_mm": (0.0, 4000.0),
    "mean_temp_c": (5.0, 45.0),
    "hot_days": (0.0, 120.0),
    "soil_awc_mm": (20.0, 300.0),
    TARGET: (0.0, 15.0),
}


def prepare_frame(df: pd.DataFrame, require_target: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Validate a table, fill documented defaults and return (clean_frame, warnings).

    Raises ValueError for missing required columns, non-numeric or out-of-range
    values. Optional columns that are absent are filled with defaults, and the
    warning list says so, because a defaulted column carries no information.
    """
    warnings: List[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    needed = REQUIRED + ([TARGET] if require_target else [])
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    for col in FEATURE_COLUMNS + [TARGET]:
        if col in out.columns:
            converted = pd.to_numeric(out[col], errors="coerce")
            bad = converted.isna() & out[col].notna()
            if bad.any():
                raise ValueError(f"Column '{col}' has non-numeric values (first at row {int(bad.idxmax()) + 2}).")
            out[col] = converted.astype(float)

    for col, default in DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
            warnings.append(f"Column '{col}' not found, using the default {default} for every row.")
        elif out[col].isna().any():
            n = int(out[col].isna().sum())
            out[col] = out[col].fillna(default)
            warnings.append(f"{n} blank value(s) in '{col}' replaced by the default {default}.")
    for col in OPTIONAL_NAN:
        if col not in out.columns:
            out[col] = np.nan

    for col in REQUIRED:
        if out[col].isna().any():
            raise ValueError(f"Column '{col}' has blank values. Remove or fill those rows.")
    if require_target and out[TARGET].isna().any():
        raise ValueError(f"Column '{TARGET}' has blank values. Remove or fill those rows.")

    for col, (lo, hi) in RANGES.items():
        if col in out.columns:
            vals = out[col].dropna()
            if ((vals < lo) | (vals > hi)).any():
                raise ValueError(f"Column '{col}' has values outside the plausible range {lo} to {hi}.")
    return out, warnings
