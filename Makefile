export PYTHONPATH := src
demo:      ## run everything on PLACEHOLDER data (to check your setup only)
	python -m hpk.synth
	python -m hpk.clean --inp data/sample/SYNTHETIC_listings_raw.csv
	python -m hpk.train
	python -m hpk.valuation
clean:     ## real data: data/raw/listings.csv -> data/processed/listings_clean.csv
	python -m hpk.clean
train:
	python -m hpk.train --trials 40
score:
	python -m hpk.valuation
sql:
	python -m hpk.db && sqlite3 data/processed/listings.db < sql/analysis.sql
app:
	streamlit run app/streamlit_app.py
test:
	pytest -q
