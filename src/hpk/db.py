"""Load the clean table into SQLite so the analysis is done in SQL, as the CV bullet claims."""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/clean_2024.csv")
    ap.add_argument("--db", default="data/processed/listings.db")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    con = sqlite3.connect(a.db)
    df.to_sql("listings", con, if_exists="replace", index=False)
    con.executescript(Path("sql/indexes.sql").read_text())
    print("loaded", len(df), "rows into", a.db)


if __name__ == "__main__":
    main()
