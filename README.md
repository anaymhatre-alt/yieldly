# Fuzzy Cotton Yield Controller

A small Python/Streamlit application demonstrating a Mamdani fuzzy logic controller specifically for cotton yield estimation. Inputs are cotton cultivation area, growing-season rainfall, and supplemental irrigation. Crop settings provide potential yield and reference levels for optimal rainfall and adequate irrigation.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The controller fuzzifies rainfall and irrigation as low, medium, or high, applies a 3×3 cotton yield IF–THEN rule base, aggregates output membership sets, and uses centroid defuzzification. It estimates yield per hectare, then multiplies it by the entered cultivated area. Irrigation can offset low rainfall; the rules reduce estimates under very high combined water conditions.

Default cotton values and rules are illustrative, not verified recommendations or predictions. Calibrate them using trusted local historical yield, weather, and irrigation data before using estimates for decisions. The model currently omits soil, temperature, cotton variety, pests, and within-region geospatial variation.
