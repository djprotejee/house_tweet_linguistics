# Requirements Coverage

This checklist maps the original project requirements to the current repository state.

## Project And Setup

| Requirement | Status | Evidence |
| --- | --- | --- |
| Project inside `F:\LNU\8sem\ATI` | Done | Project root: `F:\LNU\8sem\ATI\house_tweet_linguistics` |
| English project files and paths | Done | `README.md`, `docs/`, `src/`, `data_json/`, `data_txt/`, `metadata/` |
| PyCharm-compatible Python project | Done | `pyproject.toml`, `requirements.txt`, `.venv/`, `src/house_tweet_linguistics/` |
| README and documentation | Done | `README.md`, `docs/methodology.md`, `docs/data_management.md`, `reports/integrated_final_report.md` |

## Corpus Construction

| Requirement | Status | Evidence |
| --- | --- | --- |
| House member metadata with party, state, district, handle | Done | `metadata/accounts.csv` |
| Republican and Democratic House accounts | Done | `metadata/accounts.csv`: 217 Republican, 212 Democratic |
| Official-handle source preserved | Done | `source_url` column in `metadata/accounts.csv` |
| Tweet JSONL with raw and cleaned text | Done | `data_json/tweets.jsonl`, `data_json/tweets_extended_window.jsonl` |
| Main fixed-window authored-only corpus | Done | `data_json/tweets.jsonl` |
| Maximized 50-post corpus | Done | `data_json/tweets_extended_window.jsonl` |
| Raw API preservation | Done | `data_json/socialdata_raw_pages.jsonl` and repair raw JSONL files |
| Replacement-handle provenance | Done | `metadata/handle_repair_candidates.csv` and row-level JSON flags |

## Corpus Variants

| Variant | Status | Evidence |
| --- | --- | --- |
| Strict balanced corpus | Done | `data_txt/strict/balanced/`, `metadata/inclusion_report_strict_balanced.csv` |
| Strict full corpus | Done | `data_txt/strict/full/`, `metadata/inclusion_report_strict_full.csv` |
| Extended balanced corpus | Done | `data_txt/extended/balanced/`, `metadata/inclusion_report_extended_balanced.csv` |
| Extended full corpus | Done | `data_txt/extended/full/`, `metadata/inclusion_report_extended_full.csv` |
| Corpus count summary | Done | `metadata/corpus_variants.csv` |

## Text Mirrors

| Requirement | Status | Evidence |
| --- | --- | --- |
| Tweet-level UTF-8 text files | Done | `data_txt/<corpus>/<mode>/tweet_level/` |
| User-level merged UTF-8 text files | Done | `data_txt/<corpus>/<mode>/user_level/` |
| Party-level merged UTF-8 text files | Done | `data_txt/<corpus>/<mode>/party_level/` |

## Preprocessing

| Requirement | Status | Evidence |
| --- | --- | --- |
| Raw text preserved | Done | `raw_text` in JSONL files |
| Cleaned text preserved | Done | `cleaned_text` in JSONL files |
| No-punctuation variant preserved | Done | `cleaned_no_punct` in JSONL files |
| URL, mention, hashtag, whitespace normalization | Done | `src/house_tweet_linguistics/text.py` |
| Tokenization support | Done | `src/house_tweet_linguistics/text.py` |

## Statistical Features

| Requirement | Status | Evidence |
| --- | --- | --- |
| L in words and characters | Done | `data_tables/<corpus>_<mode>/features_by_tweet.csv`, `features_by_user.csv`, `features_by_party.csv` |
| V and V/L | Done | Same feature tables |
| Mean word length | Done | Same feature tables |
| Mean sentence length | Done | Same feature tables |
| Zipf rank-frequency tables and plots | Done | `zipf_by_party.csv`, `reports/<corpus>_<mode>/figures/zipf_*.png` |
| Heaps vocabulary growth tables and plots | Done | `heaps_by_party.csv`, `reports/<corpus>_<mode>/figures/heaps_*.png` |
| Word n-grams | Done | `ngrams_by_party.csv` |
| Character n-grams | Done | `ngrams_by_party.csv` |
| Keywords and TF-IDF support | Done | `key_terms_by_party.csv` |
| Repetition evidence | Done | `repetition_by_user.csv`, `repetition_by_party.csv`, frequency dictionaries, and n-grams |
| Stylistic feature block | Done | `style_features_by_tweet.csv`, `style_features_by_user.csv`, `style_features_by_party.csv`, `style_party_effects.csv`, `style_party_effects.png` |
| NMF topic modeling | Done | `topic_model_terms.csv`, `topic_model_by_tweet.csv`, `topic_model_by_user.csv`, `topic_model_by_party.csv`, `topic_party_effects.csv`, `topic_party_differences.png` |
| Transparent issue dictionaries | Done | `issue_dictionaries.csv`, `issue_scores_by_tweet.csv`, `issue_scores_by_user.csv`, `issue_scores_by_party.csv`, `issue_party_effects.csv`, `issue_dictionary_rates.png` |

## Distances, Clustering, Maps

| Requirement | Status | Evidence |
| --- | --- | --- |
| Cosine distance on TF-IDF | Done | `distances_by_user.csv` |
| Jensen-Shannon distance | Done | `distances_by_user.csv` |
| R-R, D-D, R-D comparison | Done | `reports/<corpus>_<mode>/distance_summary.csv` |
| Clustering | Done | `clustering_by_user.csv` |
| 2D account map | Done | `reports/<corpus>_<mode>/figures/accounts_2d_map.png` |
| Randomization control | Done | `reports/<corpus>_<mode>/randomization_report.csv` |
| Balanced vs full comparison | Done | `strict_balanced`, `strict_full`, `extended_balanced`, `extended_full` outputs |

## Extended Analysis

| Requirement | Status | Evidence |
| --- | --- | --- |
| Network analysis | Done | `network_features_by_party.csv` |
| Correlations | Done | `feature_correlations_by_user.csv`, `party_feature_effects.csv` |
| Style correlations | Done | `style_feature_correlations_by_user.csv` |
| Autocorrelations | Done | `temporal_autocorrelation_by_user.csv` |
| Party classifier control | Done | `party_classifier_summary.csv`, `party_classifier_predictions.csv`, `party_classifier_top_features.csv`, `party_classifier_confusion_matrix.png` |
| LLM module | Not included | Intentionally excluded from the main method |

## Reporting

| Requirement | Status | Evidence |
| --- | --- | --- |
| Integrated readable report | Done | `reports/integrated_final_report.md` |
| Methodology documentation | Done | `docs/methodology.md` |
| Research data management documentation | Done | `docs/data_management.md` |
| Presentation-ready figures | Done | `reports/<corpus>_<mode>/figures/` |

## Recommended Next Step

Use `reports/integrated_final_report.md` as the main interpretive report. Use `docs/methodology.md` and `docs/data_management.md` for methodological and reproducibility details.
