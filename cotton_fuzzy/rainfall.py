"""Parsing and seasonal aggregation for uploaded rainfall CSV files."""
from __future__ import annotations

import calendar
from typing import Optional

import pandas as pd


def _key(value) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def _find_column(frame: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    columns = {_key(name): name for name in frame.columns}
    return next((columns[alias] for alias in aliases if alias in columns), None)


def prepare_rainfall_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert daily, monthly-long, or year-by-month CSVs to date/rain/location.

    Tidy files need Date + rainfall_mm or Year + Month + rainfall_mm. Wide
    monthly files need Year and Jan-Dec columns. Location is optional.
    """
    date_col = _find_column(frame, ["date", "datetime", "observationdate", "recorddate"])
    year_col = _find_column(frame, ["year", "yr"])
    month_col = _find_column(frame, ["month", "monthnumber", "mnth"])
    rain_col = _find_column(frame, [
        "rainfallmm", "rainfall", "dailyrainfallmm", "monthlyrainfallmm",
        "actualrainfall", "actualrainfallmm", "rainfallactual", "precipitationmm",
    ])
    location_col = _find_column(frame, ["location", "district", "station", "subdivision", "meteorologicalsubdivision"])

    wide_month_cols = {}
    for col in frame.columns:
        key = _key(col)
        for month in range(1, 13):
            if key in {_key(calendar.month_name[month]), _key(calendar.month_abbr[month])}:
                wide_month_cols[col] = month
                break

    if rain_col is None:
        if year_col is None or not wide_month_cols:
            raise ValueError("Use Date + rainfall_mm, Year + Month + rainfall_mm, or Year with Jan-Dec columns.")
        id_columns = [year_col] + ([location_col] if location_col else [])
        long = frame.melt(id_vars=id_columns, value_vars=list(wide_month_cols), var_name="_month_name", value_name="rainfall_mm")
        long["_month"] = long["_month_name"].map(wide_month_cols)
        dates = pd.to_datetime({"year": pd.to_numeric(long[year_col], errors="coerce"), "month": long["_month"], "day": 1}, errors="coerce")
        result = pd.DataFrame({"date": dates, "rainfall_mm": pd.to_numeric(long["rainfall_mm"], errors="coerce")})
        if location_col:
            result["location"] = long[location_col].astype("string").str.strip()
    else:
        if date_col:
            dates = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True)
        elif year_col and month_col:
            years = pd.to_numeric(frame[year_col], errors="coerce")
            months = pd.to_numeric(frame[month_col], errors="coerce")
            if months.isna().any():
                month_names = {_key(calendar.month_name[m]): m for m in range(1, 13)}
                month_names.update({_key(calendar.month_abbr[m]): m for m in range(1, 13)})
                months = months.fillna(frame[month_col].map(lambda val: month_names.get(_key(val))))
            dates = pd.to_datetime({"year": years, "month": months, "day": 1}, errors="coerce")
        else:
            raise ValueError("Add a Date column, or both Year and Month columns.")
        result = pd.DataFrame({"date": dates, "rainfall_mm": pd.to_numeric(frame[rain_col], errors="coerce")})
        if location_col:
            result["location"] = frame[location_col].astype("string").str.strip()

    result = result.dropna(subset=["date", "rainfall_mm"])
    if result.empty:
        raise ValueError("No usable date and rainfall values were found in the CSV.")
    if (result["rainfall_mm"] < 0).any():
        raise ValueError("Rainfall values cannot be negative.")
    return result


def kharif_rainfall_total(frame: pd.DataFrame, year: int, start_month: int, end_month: int, location: Optional[str] = None) -> tuple[float, int]:
    """Return accumulated rainfall in the selected year, month window, location."""
    selected = frame[
        (frame["date"].dt.year == year)
        & frame["date"].dt.month.between(start_month, end_month)
    ]
    if location is not None and "location" in selected:
        selected = selected[selected["location"] == location]
    if selected.empty:
        raise ValueError("No rainfall records match the selected year, location, and season months.")
    return float(selected["rainfall_mm"].sum()), len(selected)
