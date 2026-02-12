# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NTU SCTP Capstone project analyzing ~1M Singapore job postings. The pipeline cleans raw job data, matches job titles to the government SkillsFuture taxonomy using sentence-transformer embeddings, attaches skills, and presents interactive visualizations via Streamlit.

## Commands

```bash
# Environment setup (requires uv package manager)
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync

# Run the Streamlit app
streamlit run app.py

# Lock dependencies (maintainer only)
uv lock
```

No test runner, linter, or CI/CD is configured.

## Architecture

### Data Pipeline

```
data/SGJobData.csv.xz (raw, compressed)
  → [notebooks/sgjobdata-eda.ipynb] cleaning & processing
  → data/cleaned-sgjobdata.parquet
  → data/cleaned-sgjobdata-exploded.parquet
  → data/cleaned-sgjobdata-withskills.parquet
```

### Key Processing Steps (in sgjobdata-eda.ipynb)

1. **strip_clean_drop()** — normalizes columns, handles unicode, drops empty rows
2. **winsorize_salary_log_iqr()** — caps salary outliers via log-scale IQR (bounds: 1,110–19,783)
3. **explode_categories()** — parses JSON category strings into separate rows
4. **build_title_matcher() / match_titles_embedding()** — semantic matching of job titles to SkillsFuture taxonomy using all-MiniLM-L6-v2 + fuzzy string matching (thresholds: min_sim=0.72, min_margin=0.05, min_fuzzy=85)
5. **attach_skills()** — merges matched titles with government skill taxonomy

### Key Files

- **app.py** — Streamlit entry point (stub; visualizations being ported from notebook)
- **notebooks/sgjobdata-eda.ipynb** — Main notebook producing all parquet outputs
- **notebooks/visualplayground.ipynb** — Visualization prototyping before app.py integration
- **data/jobsandskills-skillsfuture-skills-framework-dataset.xlsx** — Government skills taxonomy

### Data Schema

Cleaned dataset (1,044,597 rows): job_id, title, company, min_exp, positionlevels, posting_date, num_applications, num_views, num_vacancies, categories, average_salary_cleaned. The skills-enriched version adds jobtitle_cleaned and skill columns (6.2M rows after explosion).

## Known Data Issues

- Salary field has user-input quality problems (skipped entries, annual vs monthly confusion)
- Semantic title matching loses cultural context (e.g., "Driver" → "Engine Driver" instead of "Transport Operator")
- Earlier LLM-based skill generation was abandoned in favor of the government taxonomy
