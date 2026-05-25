# Research Data Management And Reproducibility

This document describes how research data and generated artifacts are organized in this repository.

## Data Layers

The project separates data into layers so that the corpus can be audited, regenerated, and interpreted without mixing primary records with derived artifacts.

| Layer | Location | Role |
| --- | --- | --- |
| Account metadata | `metadata/` | Member, party, district, account, inclusion, and repair metadata |
| Structured corpus | `data_json/` | Primary structured tweet records and preserved raw API pages |
| Text mirrors | `data_txt/` | Derived UTF-8 text files at tweet, account, and party levels |
| Analysis tables | `data_tables/` | Derived feature tables, distances, topics, classifier outputs |
| Figures and reports | `reports/` | Interpretive outputs and visualizations |
| Source code | `src/`, `scripts/`, `config/` | Reproducible analysis implementation |

## Primary And Derived Artifacts

The structured JSONL corpus files are the central machine-readable corpus layer:

- `data_json/tweets.jsonl`
- `data_json/tweets_extended_window.jsonl`

The text mirrors in `data_txt/` are derived from the JSONL files. They are important research outputs because they make the corpus readable and reusable in external text-analysis tools, but they are not the only source of truth.

The analysis tables in `data_tables/` and figures in `reports/<corpus>_<mode>/figures/` are derived analytical outputs. They can be regenerated from the structured corpus and source code.

Raw SocialData files in `data_json/socialdata*_raw*.jsonl` preserve collection provenance. They are useful for audit and reconstruction, but they are large and should be handled as research data rather than lightweight source files.

## Reproducibility From Existing Data

If the structured corpus files are present, text mirrors can be regenerated with:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus strict --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus strict --full
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus extended --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus extended --full
```

Analysis tables and figures can be regenerated with:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus strict --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus strict --full
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus extended --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus extended --full
```

These commands do not call external APIs. They operate on local files.

## Repository Tracking Policy

The repository should prioritize source code, documentation, metadata, and compact research reports.

Recommended to track:

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.env.example`
- `config/`
- `scripts/`
- `src/`
- `docs/`
- `reports/*.md`
- `metadata/accounts.csv`
- `metadata/corpus_variants.csv`
- `metadata/handle_repair_candidates.csv`
- `metadata/inclusion_report_*.csv`
- `data_json/tweets.jsonl`
- `data_json/tweets_extended_window.jsonl`

Recommended not to track in git:

- `.env`
- `.venv/`
- `.idea/`
- Python caches
- `logs/`
- `data_txt/`
- raw JSONL API dumps such as `data_json/socialdata_raw_pages.jsonl` and `data_json/socialdata_repair_*.jsonl`
- large generated CSV tables
- generated figures, unless a specific release needs a self-contained visual report

## Why `data_txt/` Is Not A Primary Git Artifact

The `data_txt/` directory contains the three UTF-8 mirror levels required by the research design:

- tweet level;
- account level;
- party level.

Locally, this directory contains tens of thousands of small files. It is useful for inspection and external text-processing tools, but it is fully reproducible from the structured JSONL corpus. Storing it in git would add a large number of generated files without improving the reproducibility of the analysis.

The recommended practice is to keep `data_txt/` locally and regenerate it when needed.

## Research Package Variants

### Full Local Research Package

A full local research package contains:

- source code and configuration;
- documentation;
- metadata;
- structured JSONL corpora;
- text mirrors;
- analysis tables;
- reports and figures;
- raw collection files when auditability is required.

This package is the most complete version and supports local inspection without rerunning every step.

### Lightweight Reproducible Package

A lightweight reproducible package contains:

- source code and configuration;
- documentation;
- metadata;
- structured JSONL corpora;
- main reports.

Text mirrors, analysis tables, and figures can be regenerated from this package.

### Source-Only Package

A source-only package contains:

- source code and configuration;
- documentation;
- metadata templates;
- `.env.example`;
- no corpus data.

This package supports reuse of the pipeline but does not reproduce the current study until corpus data are supplied.

## Collection Provenance

The project preserves collection provenance through metadata columns, JSON flags, raw API pages, collection logs, and handle-repair documentation.

The strict corpus should be interpreted as the primary fixed-window authored-only corpus. The extended corpus should be interpreted as a robustness corpus that tests stability under a maximized 50-post-per-member design.

## Sensitive Files

The `.env` file can contain API credentials and must not be committed. API keys, bearer tokens, paid-provider credentials, and account credentials should remain outside version control.
