import streamlit as st

from fuzzy_controller import CottonProfile, estimate_cotton_yield

st.set_page_config(page_title="Cotton Yield Controller", page_icon="🌱", layout="centered")
st.title("🌱 Cotton Yield Estimator")
st.write("Estimate cotton yield from cultivated area, growing-season rainfall, and irrigation.")
st.info("This educational fuzzy-logic demo uses illustrative cotton settings. Use locally calibrated agricultural data for planning.")

with st.sidebar:
    st.header("Cotton settings")
    potential = st.number_input("Potential cotton yield (tonnes/hectare)", min_value=0.1, value=2.5, step=0.1)
    optimum = st.number_input("Typical optimal seasonal rainfall (mm)", min_value=1.0, value=700.0, step=25.0)
    adequate_irrigation = st.number_input("Adequate supplemental irrigation reference (mm)", min_value=1.0, value=300.0, step=25.0)
    st.caption("Adjust these reference values to match trusted local data.")

left, right = st.columns(2)
with left:
    area = st.number_input("Cotton cultivation area (hectares)", min_value=0.1, value=1.0, step=0.5)
with right:
    rainfall = st.number_input("Rainfall during cotton growing season (mm)", min_value=0.0, value=float(optimum), step=25.0)
irrigation = st.number_input("Supplemental irrigation applied during the season (mm)", min_value=0.0, value=0.0, step=25.0)

if st.button("Estimate cotton yield", type="primary", use_container_width=True):
    profile = CottonProfile(
        potential_yield_t_ha=potential,
        optimal_rainfall_mm=optimum,
        adequate_irrigation_mm=adequate_irrigation,
    )
    result = estimate_cotton_yield(area, rainfall, irrigation, profile)
    st.subheader("Estimated cotton harvest")
    a, b = st.columns(2)
    a.metric("Estimated yield per hectare", f"{result['yield_per_ha_t']:.2f} t/ha")
    b.metric("Estimated total yield", f"{result['total_yield_t']:.2f} tonnes")
    st.progress(min(result["yield_fraction"], 1.0), text=f"{result['yield_fraction']:.0%} of configured cotton yield potential")
    with st.expander("How the fuzzy controller reached this estimate"):
        st.write("The controller fuzzifies rainfall and irrigation as low, medium, or high, applies cotton yield IF–THEN rules, combines the output sets, and uses centroid defuzzification.")
        st.write("Rainfall memberships", {key: f"{value:.2f}" for key, value in result["rainfall_memberships"].items()})
        st.write("Irrigation memberships", {key: f"{value:.2f}" for key, value in result["irrigation_memberships"].items()})
        st.caption("The rules allow irrigation to offset low rainfall and reduce estimated yield when both rainfall and irrigation are very high. The model does not account for soil, temperature, cotton variety, pests, or location-specific effects.")

st.divider()
st.caption("Run locally with:  pip install -r requirements.txt  ·  streamlit run app.py")
