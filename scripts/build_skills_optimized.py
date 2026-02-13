#!/usr/bin/env python3
"""Build pre-aggregated skills data for the dashboard's skills timeline chart.

Reads the withskills and exploded parquets, joins to attach category,
then aggregates by [skill, category, month_year] counting unique job_ids.

Output: data/skills_optimized.parquet
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main():
    skills = pd.read_parquet(DATA_DIR / "cleaned-sgjobdata-withskills.parquet",
                             columns=["job_id", "skill", "posting_date"])
    cats = pd.read_parquet(DATA_DIR / "cleaned-sgjobdata-exploded.parquet",
                           columns=["job_id", "category"]).drop_duplicates()

    df = skills.merge(cats, on="job_id", how="inner")
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    df = df.dropna(subset=["posting_date", "skill", "category"])
    df["month_year"] = df["posting_date"].dt.to_period("M").astype(str)

    agg = (df.groupby(["skill", "category", "month_year"])
             .agg(job_count=("job_id", "nunique"))
             .reset_index()
             .sort_values(["month_year", "category", "skill"]))

    out = DATA_DIR / "skills_optimized.parquet"
    agg.to_parquet(out, index=False, compression="snappy")
    print(f"Wrote {len(agg):,} rows to {out}")

if __name__ == "__main__":
    main()
