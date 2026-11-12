"""Streamlit front end for the fuzzy cotton yield model."""
from __future__ import annotations

import calendar

import altair as alt
import pandas as pd
import streamlit as st

from cotton_fuzzy import CottonFuzzyModel, ModelParams, Scenario
from cotton_fuzzy.rainfall import kharif_rainfall_total, prepare_rainfall_csv

st.set_page_config(page_title="Kharif Cotton Yield Fuzzy Estimator", page_icon="🌱", layout="wide")

IRRIGATION_METHODS = {"Drip (about 90%)": 0.90, "Sprinkler (about 75%)": 0.75, "Furrow or surface (about 60%)": 0.60, "Custom": None}
SOIL_TYPES = {"Light or sandy (about 70 mm)": 70.0, "Loam (about 120 mm)": 120.0, "Clay loam (about 150 mm)": 150.0,
              "Deep black clay (about 180 mm)": 180.0, "Custom": None}

with st.sidebar:
    st.header("Crop constants")
    ymax = st.number_input("Potential yield, no stress (t/ha, seed cotton)", min_value=0.3, max_value=8.0, value=2.5, step=0.1)
    etc = st.number_input("Seasonal crop water requirement (mm)", min_value=300.0, max_value=2000.0, value=900.0, step=25.0)
    reff = st.slider("Share of rainfall the crop can use", min_value=0.4, max_value=1.0, value=0.8, step=0.05)
    st.caption("Illustrative expert rules, not validated for any region. Adjust these constants to match trusted local data.")

model = CottonFuzzyModel(ModelParams(potential_yield_t_ha=float(ymax), water_requirement_mm=float(etc), rainfall_effectiveness=float(reff)))

st.title("🌱 Kharif Cotton Yield Fuzzy Estimator")
st.caption("Hierarchical Sugeno fuzzy model with water, temperature and soil rules. Rainfall can be loaded from a Kharif CSV dataset.")

rain_source = st.radio("Rainfall input", ["Upload rainfall CSV", "Enter manually"], horizontal=True)
rain = None
rain_summary = None
if rain_source == "Upload rainfall CSV":
    season_a, season_b = st.columns(2)
    with season_a:
        season_start = st.selectbox("Kharif starts", range(1, 13), index=5, format_func=lambda m: calendar.month_name[m])
    with season_b:
        season_end_options = list(range(season_start, 13))
        default_end = 9 if 9 in season_end_options else season_end_options[-1]
        season_end = st.selectbox("Kharif ends", season_end_options, index=season_end_options.index(default_end), format_func=lambda m: calendar.month_name[m])
    st.caption("Default is June–September. Adjust the months to match the local cotton calendar.")
    uploaded_rainfall = st.file_uploader("Upload daily or monthly rainfall CSV", type=["csv"])
    st.caption("Accepted formats: Date + rainfall_mm; Year + Month + rainfall_mm; or Year with Jan–Dec columns. Optional location: District, Location, Station, or Subdivision.")
    st.markdown("Official rainfall data catalog: [data.gov.in](https://data.gov.in/catalog/rainfall)")
    if uploaded_rainfall is not None:
        try:
            uploaded_rainfall.seek(0)
            rainfall_frame = prepare_rainfall_csv(pd.read_csv(uploaded_rainfall))
            available_years = sorted(rainfall_frame["date"].dt.year.unique().tolist(), reverse=True)
            selected_year = st.selectbox("Rainfall year", available_years)
            selected_location = None
            if "location" in rainfall_frame:
                locations = sorted(rainfall_frame["location"].dropna().unique().tolist())
                if locations:
                    selected_location = st.selectbox("Rainfall location", locations)
            selected = rainfall_frame[
                (rainfall_frame["date"].dt.year == selected_year)
                & rainfall_frame["date"].dt.month.between(season_start, season_end)
            ]
            if selected_location is not None:
                selected = selected[selected["location"] == selected_location]
            rain, records = kharif_rainfall_total(rainfall_frame, selected_year, season_start, season_end, selected_location)
            rain_summary = f"{rain:.1f} mm from {records} records in {selected_year}"
            st.metric("Accumulated Kharif rainfall", f"{rain:.1f} mm")
            st.caption(f"Using {records} dataset records for {calendar.month_name[season_start]}–{calendar.month_name[season_end]} {selected_year}.")
            monthly_rain = selected.groupby(selected["date"].dt.month)["rainfall_mm"].sum()
            monthly_rain.index = [calendar.month_abbr[int(month)] for month in monthly_rain.index]
            st.bar_chart(monthly_rain)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not use this CSV: {exc}")
else:
    rain = st.number_input("Kharif rainfall, whole season (mm)", min_value=0.0, max_value=5000.0, value=650.0, step=25.0)

c1, c2, c3 = st.columns(3)
area = c1.number_input("Field area (ha)", min_value=0.1, max_value=100000.0, value=1.0, step=0.5)
if rain_source == "Upload rainfall CSV":
    c2.metric("Rainfall supplied to model", f"{rain:.1f} mm" if rain is not None else "Upload a dataset")
else:
    c2.write("Rainfall entered manually above")
irr = c3.number_input("Irrigation applied, whole season (mm)", min_value=0.0, max_value=3000.0, value=200.0, step=25.0)

if rain is None:
    st.info("Upload a CSV with rainfall records, or select manual entry to continue.")
    st.stop()
if rain_summary:
    st.caption(f"Selected rainfall input: {rain_summary}.")

d1, d2 = st.columns(2)
method = d1.selectbox("Irrigation method", list(IRRIGATION_METHODS))
eta = IRRIGATION_METHODS[method]
if eta is None:
    eta = d2.slider("Share of irrigation water the crop uses", min_value=0.2, max_value=1.0, value=0.7, step=0.05)

know_crit = st.checkbox("I know the water received during flowering and boll formation",
                        help="That stage drives most of the yield response to water. If unchecked, the model assumes water arrived evenly through the season.")
crit_rain = crit_irr = None
if know_crit:
    e1, e2 = st.columns(2)
    crit_rain = e1.number_input("Rainfall in that stage (mm)", min_value=0.0, max_value=4000.0, value=250.0, step=10.0)
    crit_irr = e2.number_input("Irrigation in that stage (mm)", min_value=0.0, max_value=3000.0, value=0.0, step=10.0)

f1, f2, f3 = st.columns(3)
temp = f1.slider("Mean temperature, season (°C)", min_value=15.0, max_value=40.0, value=28.0, step=0.5)
hot = f2.slider("Hot days (max 35 °C or above) in flowering and boll stage", min_value=0.0, max_value=60.0, value=5.0, step=1.0)
soil_name = f3.selectbox("Soil water-holding capacity", list(SOIL_TYPES), index=1)
awc = SOIL_TYPES[soil_name]
if awc is None:
    awc = f3.number_input("Available water in the root zone (mm)", min_value=20.0, max_value=300.0, value=120.0, step=10.0)

scenario = Scenario(area_ha=float(area), rainfall_mm=float(rain), irrigation_mm=float(irr), irrigation_efficiency=float(eta),
                    critical_rainfall_mm=None if crit_rain is None else float(crit_rain),
                    critical_irrigation_mm=None if crit_irr is None else float(crit_irr),
                    mean_temp_c=float(temp), hot_days=float(hot), soil_awc_mm=float(awc))
est = model.estimate(scenario)

st.subheader("Estimate")
m1, m2, m3 = st.columns(3)
m1.metric("Yield per hectare", f"{est['yield_per_ha_t']:.2f} t/ha")
m2.metric("Total for the field", f"{est['total_yield_t']:.1f} t")
m3.metric("Share of potential", f"{est['yield_fraction']:.0%}")

head = est["headroom_t_ha"]
top = max(head, key=head.get)
if head[top] > 0.05 * model.params.potential_yield_t_ha:
    st.info(f"Biggest limit on this scenario: **{top.lower()}**, about {head[top]:.2f} t/ha below what ideal conditions for that factor would give.")
else:
    st.info("No single factor is limiting much in this scenario.")

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Yield headroom by factor** (t/ha gained if that factor were ideal)")
    hd = pd.DataFrame({"factor": list(head), "t_ha": list(head.values())})
    st.altair_chart(alt.Chart(hd).mark_bar().encode(x=alt.X("t_ha:Q", title="t/ha"), y=alt.Y("factor:N", sort="-x", title=None)).properties(height=150))
with g2:
    st.markdown("**What-if: change in yield (t/ha)**")
    what_if = [("Rainfall −20%", "rainfall_mm", rain * 0.8), ("Rainfall +20%", "rainfall_mm", rain * 1.2),
               ("Irrigation −100 mm", "irrigation_mm", max(0.0, irr - 100)), ("Irrigation +100 mm", "irrigation_mm", irr + 100),
               ("Temperature −3 °C", "mean_temp_c", max(15.0, temp - 3)), ("Temperature +3 °C", "mean_temp_c", min(40.0, temp + 3)),
               ("Hot days −5", "hot_days", max(0.0, hot - 5)), ("Hot days +10", "hot_days", hot + 10)]
    rows = [{"change": lab, "delta": float(model.response_curve(scenario, fld, [val])[0] - est["yield_per_ha_t"])} for lab, fld, val in what_if]
    wi = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(wi).mark_bar().encode(
        x=alt.X("delta:Q", title="Δ t/ha"), y=alt.Y("change:N", sort=[r["change"] for r in rows], title=None),
        color=alt.condition(alt.datum.delta > 0, alt.value("#2e7d32"), alt.value("#c62828"))).properties(height=230))

with st.expander("How the model reached this number"):
    a, b = st.columns(2)
    a.metric("Season water ratio (supply / requirement)", f"{est['season_water_ratio']:.2f}")
    b.metric("Flowering/boll water ratio", f"{est['critical_water_ratio']:.2f}")
    a.metric("Water index", f"{est['water_index']:.2f}")
    b.metric("Thermal index", f"{est['thermal_index']:.2f}")
    st.markdown("Membership of each input in its fuzzy sets")
    st.dataframe(pd.DataFrame(est["memberships"]).fillna(0.0).T.round(2))
    st.markdown("Rules that fired (strength is the product of the input memberships)")
    st.dataframe(est["rules"].assign(strength=lambda d: d.strength.round(3), output=lambda d: d.output.round(3)))
