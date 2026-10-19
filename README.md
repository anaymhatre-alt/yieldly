# Cotton Yield Fuzzy Estimator

A Streamlit app that estimates cotton yield with a hierarchical Sugeno fuzzy model.

## Run (Windows PowerShell)

```powershell
cd C:\projects\fuzzy_logic_cotton_yield
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Run the tests with `pip install pytest` then `python -m pytest`.

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
tests/                 pytest suite
```
