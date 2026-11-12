# Vidarbha Cotton Yield Fuzzy Estimator

A Streamlit app that estimates seed cotton yield scenarios using a hierarchical Sugeno fuzzy model. Its built-in rainfall dataset contains monthly actual rainfall for the Vidarbha meteorological subdivision, published by IMD Pune. The data file is a snapshot from 12 November 2026; check the source for newer records.

## Run (Windows PowerShell)

```powershell
cd C:\path\to\this\project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## Rainfall data

The app defaults to `vidarbha_rainfall_monthly.csv`, a historical monthly series from the [IMD Pune Vidarbha rainfall page](https://imdpune.gov.in/cmpg/subdivrainfall/vidarbha.html). The built-in data represents the Vidarbha meteorological subdivision as a whole; it is not district-level or farm-gauge data. Choose the rainfall year and Kharif months (June–September by default). The app sums the selected months, charts their totals, and supplies that seasonal rainfall to the controller.

Choose **Upload rainfall CSV** to use other daily or monthly records, or **Enter manually** when no dataset is available. Accepted CSV formats:

```csv
date,rainfall_mm,location
2025-06-01,4.2,Akola
2025-06-02,0.0,Akola
```

You can also use `year,month,rainfall_mm,district`, or a wide monthly file with a `year` column and `Jan` through `Dec` columns. The year-by-month rainfall needs months represented as month names or numbers. Daily and monthly rainfall inputs are both accumulated within the selected season.

## Model and limits

- **Water** rules combine seasonal and flowering/boll-stage water into a water index.
- **Thermal** rules use mean temperature and hot days for a thermal index.
- **Final** rules combine water, thermal conditions, and soil water-holding capacity into a fraction of potential yield.
- The app multiplies yield per hectare by field area to estimate total seed cotton.

Fuzzy rule values and default crop constants are illustrative and have not been validated against Vidarbha cotton trial or farm records. The built-in regional rainfall series does not represent individual districts or farms. Treat results as educational scenario estimates, not field forecasts. Variety, pests, disease, nutrients, sowing date, and within-field variation are not modeled.

## Project files

```text
app.py                         Streamlit user interface
cotton_fuzzy/                  Fuzzy memberships, rules, and model
vidarbha_rainfall_monthly.csv  IMD Pune Vidarbha monthly actual rainfall snapshot
rainfall_data_template.csv     Blank daily rainfall upload template
requirements.txt               Python dependencies
```
