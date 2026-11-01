# Project 3 plan (about 10 to 14 days)

| Day | Goal | Done when |
|---|---|---|
| 1 | `pip install`, `make test`, `make demo` | demo runs, app opens |
| 2 | Pick data source; read ToS + robots.txt; decide scrape vs public dataset | decision written in README |
| 3-6 | Collect 40k+ listings (4 cities, slowly, cached) or ingest public set | `data/raw/listings.csv` |
| 7 | `make clean`; read `cleaning_report.json`; eyeball 50 random rows by hand | parsers verified on real strings |
| 8 | `make train` (30-50 Optuna trials); fill the README results table | metrics.csv real |
| 9 | Error analysis: which city / price band is worst, and why | 1 paragraph in README |
| 10 | `make score`; manually check 10 flagged "undervalued" listings | note which flags were wrong |
| 11 | `make sql`; save 3 insights with numbers | insights in README |
| 12-13 | Deploy app on Streamlit Community Cloud (commit `models/*.joblib` + `area_stats.csv`, not raw data) | public link |
| 14 | Record README screenshots, push to GitHub, write CV bullet | done |

## CV bullet (fill the X's from reports/ after real data only)
"Collected and cleaned [N] property listings across 4 Pakistani cities (normalising marla/kanal/sq yd areas and Crore/Lakh prices);
XGBoost achieved R² of [..] and MAE of Rs [..] lakh, [..]% lower RMSE than a linear baseline; used SHAP to explain price drivers
and deployed a Streamlit price estimator with an undervalued-listings finder."

## Interview questions this project invites
- Why log-price? Why per-city outlier trimming? Why out-of-fold predictions for the undervalued finder?
- What is the marla ambiguity and how did you handle it?
- What would change if you had actual sale prices?
