from __future__ import annotations

import re
import string
from collections import Counter
from math import isnan

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .text import MENTION_RE, URL_RE, HASHTAG_RE, split_sentences, tokenize_words

EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]",
    flags=re.UNICODE,
)

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|#?[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")
CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
MONEY_RE = re.compile(r"(?:\$|usd\b|dollars?\b|million\b|billion\b)", flags=re.IGNORECASE)

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
NEGATIONS = {"no", "not", "never", "none", "nothing", "neither", "nor", "without", "cannot", "can't", "dont", "don't", "wont", "won't"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would", "need", "needs", "needed"}

POSITIVE_WORDS = {
    "achieve", "approve", "benefit", "bipartisan", "celebrate", "champion", "congratulations",
    "deliver", "effective", "excellent", "good", "great", "honor", "improve", "invest",
    "opportunity", "proud", "safe", "secure", "strengthen", "support", "thank", "thanks",
    "win", "wins", "working",
}
NEGATIVE_WORDS = {
    "attack", "bad", "broken", "crisis", "dangerous", "deadly", "deficit", "disaster",
    "fail", "failed", "failure", "fraud", "harm", "illegal", "inflation", "lawless",
    "lie", "lies", "radical", "shame", "threat", "waste", "wrong",
}
ANGER_WORDS = {
    "attack", "corrupt", "disgrace", "furious", "outrage", "outrageous", "radical",
    "shame", "shameful", "slammed", "unacceptable",
}
RISK_WORDS = {
    "border", "crisis", "danger", "dangerous", "emergency", "risk", "secure", "security",
    "threat", "unsafe", "violence",
}


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _per_100(value: float, words: int) -> float:
    return _safe_divide(value * 100.0, words)


def _syllable_count(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    groups = re.findall(r"[aeiouy]+", cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _cliffs_delta(left: pd.Series, right: pd.Series) -> float:
    left_values = left.dropna().to_numpy(dtype=float)
    right_values = right.dropna().to_numpy(dtype=float)
    if len(left_values) == 0 or len(right_values) == 0:
        return 0.0
    greater = 0
    lower = 0
    for value in left_values:
        greater += int(np.sum(value > right_values))
        lower += int(np.sum(value < right_values))
    return float((greater - lower) / (len(left_values) * len(right_values)))


def style_metrics(raw_text: str, cleaned_text: str | None = None) -> dict[str, float | int]:
    text = str(raw_text or "")
    cleaned = str(cleaned_text if cleaned_text is not None else raw_text or "")
    words_raw = WORD_RE.findall(text)
    words = tokenize_words(cleaned)
    word_count = len(words)
    sentences = split_sentences(text)
    sentence_count = len(sentences)
    syllables = sum(_syllable_count(word) for word in words)
    chars = len(text)
    letters = sum(1 for char in text if char.isalpha())
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    caps_words = CAPS_WORD_RE.findall(text)
    lower_counter = Counter(token.lower().lstrip("#") for token in words)

    flesch_reading_ease = 206.835 - 1.015 * _safe_divide(word_count, sentence_count) - 84.6 * _safe_divide(syllables, word_count)
    flesch_kincaid_grade = 0.39 * _safe_divide(word_count, sentence_count) + 11.8 * _safe_divide(syllables, word_count) - 15.59

    result = {
        "raw_char_count": chars,
        "raw_letter_count": letters,
        "raw_word_count": len(words_raw),
        "clean_word_count": word_count,
        "sentence_count": sentence_count,
        "avg_raw_word_len": _safe_divide(sum(len(word) for word in words_raw), len(words_raw)),
        "url_count": len(URL_RE.findall(text)),
        "mention_count": len(MENTION_RE.findall(text)),
        "hashtag_count": len(HASHTAG_RE.findall(text)),
        "emoji_count": len(EMOJI_RE.findall(text)),
        "number_count": len(NUMBER_RE.findall(text)),
        "money_marker_count": len(MONEY_RE.findall(text)),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "ellipsis_count": text.count("...") + text.count("\u2026"),
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "colon_semicolon_count": text.count(":") + text.count(";"),
        "quote_count": text.count('"') + text.count("'") + text.count("\u201c") + text.count("\u201d"),
        "dash_count": text.count("-") + text.count("\u2013") + text.count("\u2014"),
        "newline_count": text.count("\n"),
        "punctuation_count": punctuation_count,
        "punctuation_per_100_words": _per_100(punctuation_count, word_count),
        "exclamation_per_100_words": _per_100(text.count("!"), word_count),
        "question_per_100_words": _per_100(text.count("?"), word_count),
        "hashtag_per_100_words": _per_100(len(HASHTAG_RE.findall(text)), word_count),
        "mention_per_100_words": _per_100(len(MENTION_RE.findall(text)), word_count),
        "url_per_100_words": _per_100(len(URL_RE.findall(text)), word_count),
        "emoji_per_100_words": _per_100(len(EMOJI_RE.findall(text)), word_count),
        "caps_word_count": len(caps_words),
        "caps_word_share": _safe_divide(len(caps_words), len(words_raw)),
        "uppercase_char_share": _safe_divide(sum(1 for char in text if char.isupper()), letters),
        "first_person_singular_count": sum(lower_counter[word] for word in FIRST_PERSON_SINGULAR),
        "first_person_plural_count": sum(lower_counter[word] for word in FIRST_PERSON_PLURAL),
        "second_person_count": sum(lower_counter[word] for word in SECOND_PERSON),
        "negation_count": sum(lower_counter[word] for word in NEGATIONS),
        "modal_count": sum(lower_counter[word] for word in MODALS),
        "positive_lexicon_count": sum(lower_counter[word] for word in POSITIVE_WORDS),
        "negative_lexicon_count": sum(lower_counter[word] for word in NEGATIVE_WORDS),
        "anger_lexicon_count": sum(lower_counter[word] for word in ANGER_WORDS),
        "risk_lexicon_count": sum(lower_counter[word] for word in RISK_WORDS),
        "first_person_plural_per_100_words": _per_100(sum(lower_counter[word] for word in FIRST_PERSON_PLURAL), word_count),
        "second_person_per_100_words": _per_100(sum(lower_counter[word] for word in SECOND_PERSON), word_count),
        "negation_per_100_words": _per_100(sum(lower_counter[word] for word in NEGATIONS), word_count),
        "modal_per_100_words": _per_100(sum(lower_counter[word] for word in MODALS), word_count),
        "positive_lexicon_per_100_words": _per_100(sum(lower_counter[word] for word in POSITIVE_WORDS), word_count),
        "negative_lexicon_per_100_words": _per_100(sum(lower_counter[word] for word in NEGATIVE_WORDS), word_count),
        "anger_lexicon_per_100_words": _per_100(sum(lower_counter[word] for word in ANGER_WORDS), word_count),
        "risk_lexicon_per_100_words": _per_100(sum(lower_counter[word] for word in RISK_WORDS), word_count),
        "lexicon_sentiment_balance": _per_100(
            sum(lower_counter[word] for word in POSITIVE_WORDS) - sum(lower_counter[word] for word in NEGATIVE_WORDS),
            word_count,
        ),
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
    }
    return result


STYLE_NUMERIC_COLUMNS = [
    "raw_char_count",
    "raw_word_count",
    "avg_raw_word_len",
    "url_count",
    "mention_count",
    "hashtag_count",
    "emoji_count",
    "number_count",
    "money_marker_count",
    "exclamation_count",
    "question_count",
    "punctuation_per_100_words",
    "exclamation_per_100_words",
    "question_per_100_words",
    "hashtag_per_100_words",
    "mention_per_100_words",
    "url_per_100_words",
    "emoji_per_100_words",
    "caps_word_share",
    "uppercase_char_share",
    "first_person_plural_per_100_words",
    "second_person_per_100_words",
    "negation_per_100_words",
    "modal_per_100_words",
    "positive_lexicon_per_100_words",
    "negative_lexicon_per_100_words",
    "anger_lexicon_per_100_words",
    "risk_lexicon_per_100_words",
    "lexicon_sentiment_balance",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
]


def summarize_style_effects(by_user: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = by_user[by_user["party"].isin(["Democratic", "Republican"])].copy()
    for column in STYLE_NUMERIC_COLUMNS:
        if column not in main.columns:
            continue
        democratic = main[main["party"].eq("Democratic")][column].astype(float)
        republican = main[main["party"].eq("Republican")][column].astype(float)
        if len(democratic.dropna()) == 0 or len(republican.dropna()) == 0:
            continue
        try:
            _, p_value = mannwhitneyu(republican.dropna(), democratic.dropna(), alternative="two-sided")
        except ValueError:
            p_value = np.nan
        rows.append(
            {
                "feature": column,
                "democratic_mean": float(democratic.mean()),
                "republican_mean": float(republican.mean()),
                "republican_minus_democratic": float(republican.mean() - democratic.mean()),
                "democratic_median": float(democratic.median()),
                "republican_median": float(republican.median()),
                "mann_whitney_u_p_value": float(p_value) if not isnan(float(p_value)) else np.nan,
                "cliffs_delta_republican_vs_democratic": _cliffs_delta(republican, democratic),
            }
        )
    return pd.DataFrame(rows).sort_values("mann_whitney_u_p_value", na_position="last")
