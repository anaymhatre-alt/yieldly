# Kharif Cotton Yield Fuzzy Estimator

A Streamlit app that estimates cotton yield with a hierarchical Sugeno fuzzy model. Rainfall can be entered manually or accumulated from an uploaded daily or monthly CSV for a selected Kharif period, year, and location.

## Run (Windows PowerShell)

```powershell
cd C:\projects\fuzzy_logic_cotton_yield
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Run the model tests with `pip install pytest` then `python -m pytest`.

## Rainfall CSV input

Choose **Upload rainfall CSV** in the app. The default season window is June–September, matching the India Meteorological Department's southwest monsoon window; change the months to match the local cotton calendar. The app filters by year and optional location, sums the rainfall values in the selected months, charts monthly rainfall totals, and sends the seasonal total to the model. Choose **Enter manually** when no dataset is available.

Accepted daily or monthly-long layout:

```csv
date,rainfall_mm,location
2025-06-01,4.2,Akola
2025-06-02,0.0,Akola
```

Or use `year,month,rainfall_mm,district`. Monthly wide files with a `year` column and `Jan` through `Dec` columns are also supported. Location columns are optional; recognized names include `location`, `district`, `station`, and `subdivision`. Rainfall columns include `rainfall_mm`, `rainfall`, `actual_rainfall`, and common daily/monthly variants. See `rainfall_data_template.csv` for a blank daily-data template.

The [Open Government Data rainfall catalog](https://data.gov.in/catalog/rainfall) lists Indian district-wise and other rainfall datasets. Choose a dataset for the relevant location and check its units and coverage before uploading. The app does not download rainfall data automatically. See the [IMD monsoon information](https://mausam.imd.gov.in/imd_latest/contents/hydrological-services.php) for the June–September southwest monsoon context.

## Model

Two first-level modules feed a final module. Every rule outputs a number between 0 and 1.

- **Water** (season water ratio, flowering/boll water ratio) gives a water index.
- **Thermal** (mean temperature, hot days) gives a thermal index.
- **Final** (water index, thermal index, soil water capacity) gives a fraction of potential yield, which is multiplied by the potential yield and the field area.

Memberships are overlapping triangles whose outer sets extend to infinity, so no input value falls outside the model and the output has no jumps.

The rule values are illustrative expert guesses, not validated for any region. Not modelled: variety, pests and disease, nutrients, sowing date, within-field variation.

## Project layout

```
app.py                 Streamlit page
cotton_fuzzy/
  membership.py        overlapping fuzzy partitions
  priors.py            rule values
  model.py             the Sugeno model and scenario estimates
  data.py              input validation and defaults
  rainfall.py          CSV parsing and Kharif rainfall aggregation
rainfall_data_template.csv  blank rainfall CSV template
tests/                 pytest suite
```
