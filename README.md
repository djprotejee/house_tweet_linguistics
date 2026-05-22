# House Tweet Linguistics

Comparative statistical-linguistic study of authored posts from official X/Twitter accounts of Republican and Democratic members of the U.S. House of Representatives.

## Research Aim

The project builds Republican and Democratic House tweet corpora, creates UTF-8 text mirrors at tweet, account, and party levels, and compares the corpora with classical corpus-linguistic and statistical-linguistic methods.

Main hypothesis: if party affiliation is reflected in texts, distances between accounts from different parties should be higher on average than distances within the same party, and clustering should partially recover the Republican/Democratic split.

Null hypothesis: party labels are not associated with statistical text features, and the observed structure is not stronger than randomized labels.

## Main Documents

- `reports/integrated_final_report.md`: main readable research report with abstract, diagrams, tables, figures, interpretation, and limitations.
- `docs/methodology.md`: corpus rules, preprocessing, analysis modules, and interpretation logic.
- `docs/data_management.md`: research data layers, reproducibility, generated artifacts, and repository tracking policy.
- `reports/requirements_coverage.md`: traceability checklist mapping requirements to outputs.

## Verified Sources

- The course allows research-oriented individual work using programs on a broader range of texts: local file `../лист_старости2026corrected.pdf`, page 2.
- The course program includes corpus linguistics, statistical linguistics, Zipf, Heaps, n-grams, repetition, text distances, clustering, correlations, and networks: local file `../program_CL&NLP(new&universal).pdf`, pages 1-4.
- The Clerk of the House official list dated May 1, 2026 states 217 Republicans, 212 Democrats, 1 Independent, and 5 vacancies: https://clerk.house.gov/member_info/olm-119.pdf, page 1.
- X API user posts endpoint is `GET /2/users/:id/tweets`: https://docs.x.com/x-api/posts/timelines/introduction.
- X API user lookup supports looking up up to 100 users at once: https://docs.x.com/x-api/users/lookup/quickstart/user-lookup.
- X API rate limits list `GET /2/users/:id/tweets` as 10,000/15 min per app and 900/15 min per user: https://docs.x.com/x-api/fundamentals/rate-limits.

## Project Structure

```text
house_tweet_linguistics/
  config/                  Configuration defaults
  docs/                    Methodology and data-management documentation
  metadata/                Account, inclusion, corpus, and repair metadata
  data_json/               Structured corpus and raw collection files
  data_txt/                Generated UTF-8 text mirrors
  data_tables/             Generated analytical CSV tables
  reports/                 Research reports and generated figures
  scripts/                 Convenience scripts
  src/house_tweet_linguistics/
```

## Corpus Variants

| Corpus | JSONL file | Role |
| --- | --- | --- |
| `strict` | `data_json/tweets.jsonl` | Primary fixed-window authored-only corpus |
| `extended` | `data_json/tweets_extended_window.jsonl` | Maximized 50-post-per-member robustness corpus |

Current corpus counts are stored in `metadata/corpus_variants.csv`.

Strict corpus:

- Balanced strict subset: 205 Democratic accounts + 205 Republican accounts, 20,479 posts.
- Full strict eligible subset: 417 accounts, 20,798 posts.
- Strict JSONL total before eligibility filtering: 20,838 posts.

Extended corpus:

- Balanced extended subset: 212 Democratic accounts + 212 Republican accounts, 21,200 posts.
- Full extended subset: 429 accounts, 21,450 posts.
- Extended JSONL total: 21,450 posts.

The strict corpus is the main evidence base. The extended corpus is a robustness corpus and includes documented supplemental records and handle repairs.

## Setup

Open this directory as the project root in PyCharm or another Python IDE. The project root is the folder that contains `README.md`, `pyproject.toml`, `requirements.txt`, `src/`, `metadata/`, `data_json/`, `data_txt/`, `data_tables/`, and `reports/`.

If `.venv/` already exists, select this interpreter:

```text
.venv\Scripts\python.exe
```

If the virtual environment does not exist or must be recreated, use Python 3.10+:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
```

For analysis-only use, no API key is required if `data_json/tweets.jsonl` and `data_json/tweets_extended_window.jsonl` are already present.

For new data collection, copy `.env.example` to `.env` and add either an X API bearer token or a SocialData API key. Do not commit `.env`.

## Workflow A: Reproduce From Existing Data

Use this workflow when the JSONL corpus files are already present. It does not call external APIs.

Build UTF-8 text mirrors:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus strict --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus strict --full
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus extended --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --corpus extended --full
```

Run the full statistical-linguistic analysis:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus strict --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus strict --full
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus extended --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --corpus extended --full
```

Read the main output:

```text
reports/integrated_final_report.md
```

## Workflow B: Collect New Data

Import the House Press Gallery official X handles:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics import-handles
```

Audit account metadata:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics audit-accounts
```

Resolve usernames to user IDs:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics collect --resolve-users
```

Estimate SocialData collection cost:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics collect --estimate-socialdata-cost
```

Collect posts with SocialData using an explicit budget cap:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics collect --fetch-socialdata --budget-usd 0.25 --max-accounts 20
```

Collect posts with the configured X API flow:

```powershell
.\.venv\Scripts\python.exe -m house_tweet_linguistics collect --fetch-tweets
```

After collection, use Workflow A to rebuild mirrors and rerun analysis.

## Main Analysis Modules

The pipeline includes:

- corpus construction and three text mirrors;
- preprocessing with raw, cleaned, and no-punctuation text;
- basic statistical features `L`, `V`, `V/L`, average word length, and sentence length;
- Zipf and Heaps law tables and plots;
- word and character n-grams;
- TF-IDF and smoothed log-ratio keywords;
- repetition and lexical entropy;
- cosine and Jensen-Shannon account distances;
- R-R, D-D, and R-D distance comparison;
- clustering and 2D account maps;
- randomized-label control;
- word-adjacency network features;
- stylometric features;
- NMF topic modeling;
- transparent issue dictionaries;
- global account-level TF-IDF party-classifier control.

Methodological details are documented in `docs/methodology.md`.

## Data And Reproducibility

The project distinguishes primary structured corpus files from derived text mirrors, analysis tables, and figures.

The main structured corpus files are:

- `data_json/tweets.jsonl`
- `data_json/tweets_extended_window.jsonl`

The `data_txt/`, `data_tables/`, and figure directories are generated artifacts. They are part of the local research workspace and can be regenerated from the structured corpus files.

Data-management details are documented in `docs/data_management.md`.

## Main Conclusion

The main result is that party affiliation is reflected in aggregate statistical-linguistic distance patterns, thematic distributions, issue-marker rates, classifier recoverability, and several stylometric markers. The effect is stable across strict and extended corpus variants, but it is diffuse and coexists with substantial within-party variation rather than producing a clean two-cluster separation.
