# Methodology

This document describes the research design, corpus construction rules, preprocessing decisions, and analysis modules used in the project.

## Research Question

The project asks whether party affiliation is reflected in the statistical-linguistic properties of authored posts from official X/Twitter accounts of Republican and Democratic members of the U.S. House of Representatives.

The main hypothesis is that account-level texts from different parties should be farther apart, on average, than texts from accounts within the same party. A secondary expectation is that clustering may partially recover the Republican/Democratic split.

The null hypothesis is that party labels are not associated with the measured text features and that the observed structure is not stronger than randomized party labels.

## Corpus Scope

The unit of analysis is the authored post text associated with official or sufficiently documented X/Twitter accounts of voting members of the U.S. House of Representatives.

The main party comparison uses Republican and Democratic accounts. The Independent account is retained only as an exploratory out-of-sample case in full-corpus outputs and should not be treated as a third balanced party group.

## Authored-Only Rule

The primary corpora are authored-only corpora.

Retweets and reposts are excluded because they are not the member's own text. Replies are excluded by default because they are context-dependent and often depend on missing conversational context. Quote tweets are retained only when the returned timeline record contains the author's own text; quoted external text is not merged into the member's corpus.

This rule makes the main corpus better suited for authorship-oriented statistical-linguistic comparison. A future robustness corpus may include replies, retweets/reposts, and quote tweets as account timeline behavior, but that corpus should be interpreted separately from authored language.

## Corpus Variants

The project maintains two authored-only corpus variants.

| Corpus | JSONL file | Function |
| --- | --- | --- |
| `strict` | `data_json/tweets.jsonl` | Primary one-year evidence base |
| `extended` | `data_json/tweets_extended_window.jsonl` | Maximized robustness corpus |

### Strict Corpus

The strict corpus is the main evidence base. It uses the fixed study window from `2025-01-03T00:00:00Z` to `2026-01-03T00:00:00Z`, excludes replies and retweets, and uses the collected House account handles.

Strict outputs include:

- `data_txt/strict/balanced/`
- `data_txt/strict/full/`
- `metadata/inclusion_report_strict_balanced.csv`
- `metadata/inclusion_report_strict_full.csv`
- `data_tables/strict_balanced/`
- `data_tables/strict_full/`
- `reports/strict_balanced/`
- `reports/strict_full/`

### Extended Corpus

The extended corpus is a maximized robustness corpus with 50 posts for each voting member. It should not replace the strict corpus. It contains supplemental records outside the strict study window and documented handle repairs.

Supplemental records are marked in JSON flags, including:

- `supplemental_extended_window`
- `original_collection_window`
- `raw_source_file`
- `handle_repair_replacement_username`
- `handle_repair_reason`

The handle repair map is stored in `metadata/handle_repair_candidates.csv`.

Extended outputs include:

- `data_txt/extended/balanced/`
- `data_txt/extended/full/`
- `metadata/inclusion_report_extended_balanced.csv`
- `metadata/inclusion_report_extended_full.csv`
- `data_tables/extended_balanced/`
- `data_tables/extended_full/`
- `reports/extended_balanced/`
- `reports/extended_full/`

## Balanced And Full Modes

Each corpus has two analysis modes.

The balanced mode uses equal numbers of Democratic and Republican accounts. It is the main comparison mode because it reduces party-size imbalance.

The full mode uses all eligible accounts in the corpus. It is a robustness check that tests whether the balanced result depends on account selection.

The recommended interpretation order is:

1. Strict balanced as the primary result.
2. Strict full as a first robustness check.
3. Extended balanced as a maximized balanced robustness check.
4. Extended full as a maximized full robustness check.

## Text Mirrors

The project builds three UTF-8 text mirrors for each corpus and mode:

- tweet-level files: one file per post;
- user-level files: one merged file per account;
- party-level files: one merged file per party.

Each mirror directory also includes `account_codebook.csv` and `tweet_manifest.csv`. The account codebook maps internal account codes such as `dem004` or `rep078` to the politician's name, party, state, district, and username. The tweet manifest maps every tweet-level text file to its account code, politician, tweet ID, and creation timestamp.

The text mirrors make the corpus inspectable and reusable for external tools, but they are derived from the structured JSONL corpus files.

## Preprocessing

Raw text is preserved in the JSONL files. Cleaned text is created for analysis.

The preprocessing pipeline:

- normalizes whitespace;
- replaces URLs with a URL marker;
- replaces mentions with a MENTION marker;
- keeps hashtags as lexical tokens unless a specific function removes punctuation;
- lowercases text for frequency-based analysis;
- creates a no-punctuation variant;
- tokenizes words, sentences, character n-grams, and word n-grams.

## Analysis Modules

### Basic Statistical Features

The project computes word length `L`, character length, vocabulary size `V`, `V/L`, average word length, sentence count, average sentence length, and hapax counts at tweet, account, and party levels.

### Frequency Laws

Zipf rank-frequency tables and Heaps vocabulary-growth tables are produced for party corpora. Log-log slopes are fitted to compare broad corpus behavior across parties.

### N-Grams And Keywords

The project computes word 1-grams, 2-grams, and 3-grams, character 3-grams, 4-grams, and 5-grams, TF-IDF-supported key terms, and smoothed log-ratio party-specific terms.

### Distances And Clustering

Account-level texts are vectorized with TF-IDF. Pairwise cosine distances are computed for Republican-Republican, Democratic-Democratic, and Republican-Democratic account pairs. Jensen-Shannon distances are also computed from smoothed word-frequency distributions.

Agglomerative clustering with two clusters is used as an unsupervised check. The agreement between cluster labels and party labels is measured with Adjusted Rand Index. A 2D account map is built with Truncated SVD on the TF-IDF matrix.

### Randomization Control

Party labels are shuffled 1,000 times. For each shuffle, the between-party minus within-party cosine-distance gap is recomputed. The observed gap is then compared against the shuffled-label distribution.

### Repetition And Networks

Repetition is measured through hapax counts, repeated-token share, top-token share, and lexical entropy. Party-level word-adjacency networks are built from adjacent tokens, and basic network characteristics are computed.

### Stylometric Features

The stylometric block measures surface-form and discourse markers: tweet length, punctuation density, exclamation and question marks, uppercase share, hashtags, mentions, URLs, emojis, numbers, money markers, pronouns, negations, modals, and small transparent lexicons for positive, negative, anger, and risk vocabulary.

These features are interpreted as stylometric markers, not as a full psychological or sentiment model.

### NMF Topic Modeling

NMF topic modeling extracts unsupervised components from TF-IDF features. Topic shares are computed at tweet level and aggregated to accounts and parties. Topic labels are descriptive strings based on top-weighted terms, not manually validated categories.

### Issue Dictionaries

Transparent issue dictionaries count pre-defined issue markers per 1,000 words. Current issue groups include economy and taxes, immigration and border, health care, foreign policy and Ukraine, energy and climate, education, crime and public safety, civil rights, military and veterans, government shutdown, Trump/Biden, and agriculture/food.

Issue dictionaries are marker lists, not full semantic annotation.

### Global Party Classifier Control

The party classifier is a separate global control. It uses account-level TF-IDF features from the full account text and Logistic Regression to test whether party labels are recoverable from overall text.

It is not limited to NMF topics or issue dictionaries, and it is not treated as the main explanatory method.

## Main Reporting File

The main interpretive report is `reports/integrated_final_report.md`. It contains the abstract, design summary, result tables, Mermaid diagrams, figures, synthesis, limitations, and final conclusion.
