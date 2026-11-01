# Munasib Daam: house price estimates for Pakistan

Estimates the asking price of houses, flats and portions in 9 Pakistani cities, explains what drives the price,
and flags listings priced far from comparable ones. Built on 37,042 cleaned listings from May 2024.

- **Live app:** _TODO: add your Streamlit link_
- **Screenshot:** _TODO: add `docs/app.png`_

## Results

Held-out test set (20% of listings the model never saw, stratified by city). Model chosen by cross-validation on the training set only.

| Model | R² | Median error | MAE (Rs lakh) | RMSE (Rs lakh) |
|---|---|---|---|---|
| Linear Regression (baseline) | 0.903 | 16.1% | 109.6 | 259.9 |
| Random Forest | 0.909 | 13.7% | 99.2 | 251.7 |
| **XGBoost (Optuna, 30 trials)** | **0.920** | **13.7%** | **95.5** | **236.7** |

XGBoost has 12.8% lower MAE than the linear baseline. "Median error" is the typical gap between estimate and asking price as a share of the asking price.

| City | Test listings | R² | Median error |
|---|---|---|---|
| Islamabad | 1,606 | 0.92 | 12.8% |
| Karachi | 2,278 | 0.90 | 15.5% |
| Lahore | 2,249 | 0.89 | 13.0% |
| Rawalpindi | 629 | 0.88 | 14.0% |
| Peshawar | 117 | 0.86 | 13.6% |
| Faisalabad | 160 | 0.89 | 14.0% |
| Gujranwala | 97 | 0.90 | 11.0% |
| Multan | 227 | 0.76 | 11.2% |
| Hyderabad | 46 | 0.89 | 17.6% |

Small cities have few test listings, so their numbers move around. Multan's low R² comes from a few very large houses with large rupee errors; its median error is normal.

## What I found in the data

The raw file had 155,021 listings. Cleaning is in `src/hpk/clean.py`; the full count of what each step removed is saved to `reports/cleaning_report_2024.json`.

- **The source file's own square-foot column was wrong for every Karachi listing in square yards** (it copied the yard number, 9 times too small). All sizes are re-parsed from the ad text instead.
- **Over half the file was not houses:** plots, shops, offices and land (91,144 rows) were dropped.
- **22,803 repeat ads** for the same property were removed, so the same listing cannot sit in both train and test.
- **2,131 instalment and payment-plan ads** were removed: their price is a down payment, not the full price.
- Plot ads wrongly typed as houses, 0 bedrooms treated as missing, and per-city outliers were handled.

## How it works

1. **Parse and clean** (`parsing.py`, `clean.py`): crore/lakh prices, marla/kanal/sq yd areas, de-duplication, outliers per city.
2. **Features** (`modeling.py`, `textfeats.py`): city, area, phase/block/sector and keyword flags read from the ad URL (brand new, corner, luxury, furnished, park facing, main road, double storey, basement, renovated, leased). No price words are used, so there is no target leakage.
3. **Models** (`train.py`): predicts log price; Linear Regression vs Random Forest vs tuned XGBoost.
4. **Explainability** (`explain.py`): SHAP contributions grouped back to area, size, rooms and flags; shown in the app.
5. **Finder** (`valuation.py`): every listing is scored by a model that never saw it (5-fold out-of-fold). Listed below 70% of the estimate = undervalued, above 140% = overpriced, and gaps beyond 55% below or 2.2x above are marked **Check data** (probably an error). Areas with fewer than 30 listings are marked low confidence.
6. **App** (`app/streamlit_app.py`): estimate with a likely range, asking-price check, price drivers, area comparison, four colour themes.

## Run it (Windows PowerShell)

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q                                    # 79 tests

python -m hpk.ingest --inp data/raw/zameen.csv --out data/raw/listings_2024.csv
python -m hpk.clean --inp data/raw/listings_2024.csv --out data/processed/clean_2024.csv --report reports/cleaning_report_2024.json
python -m hpk.train --trials 30
python -m hpk.valuation
streamlit run app/streamlit_app.py
```

The scripts `python -m hpk.db` and `sql/analysis.sql` load the clean data into SQLite for price-per-sq-ft queries by city, area and size band.

## Data

- **Source:** _TODO: Kaggle dataset name, author, link and license for `zameen.csv`_.
- Public listings from Zameen.com collected in one week (20 to 26 May 2024). I did not scrape the data myself.
- Raw data is not in this repository. Place the file at `data/raw/zameen.csv`.

## Limits

- **Asking prices, not sale prices.** Final prices are usually lower.
- **One week in May 2024.** No trend analysis, and prices have moved since.
- **Condition, floor level, lease title and unfinished ("grey structure") houses are invisible to the model.** The finder's top picks are candidates to verify, not confirmed bargains.
- **Within the same area and size, asking prices vary by 2x or more**, which caps the accuracy any model can reach on this data.
- Peshawar has 584 clean listings, so its estimates are less reliable than Lahore's or Karachi's.
- A statistical estimate, not a valuation or financial advice.
