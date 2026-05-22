from __future__ import annotations

import re
import string
from collections import Counter
from math import log

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|#?[a-z][a-z0-9_]*|\d+(?:\.\d+)?", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("#", ""))


def normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def clean_text(text: str, keep_hashtags: bool = True) -> str:
    cleaned = URL_RE.sub(" URL ", text)
    cleaned = MENTION_RE.sub(" MENTION ", cleaned)
    if keep_hashtags:
        cleaned = HASHTAG_RE.sub(r"#\1", cleaned)
    else:
        cleaned = HASHTAG_RE.sub(r"\1", cleaned)
    cleaned = cleaned.lower()
    cleaned = normalize_spaces(cleaned)
    return cleaned


def remove_punctuation(text: str) -> str:
    return normalize_spaces(text.translate(PUNCT_TABLE))


def tokenize_words(text: str, remove_special_tokens: bool = True) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if remove_special_tokens:
        tokens = [token for token in tokens if token not in {"url", "mention"}]
    return tokens


def split_sentences(text: str) -> list[str]:
    sentences = [normalize_spaces(match.group(0)) for match in SENTENCE_RE.finditer(text)]
    return [sentence for sentence in sentences if sentence]


def text_metrics(text: str) -> dict[str, float | int]:
    cleaned = clean_text(text)
    tokens = tokenize_words(remove_punctuation(cleaned))
    sentences = split_sentences(text)
    chars_no_spaces = len(re.sub(r"\s+", "", text))
    vocab = set(tokens)
    word_lengths = [len(token.lstrip("#")) for token in tokens if token]
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    l_words = len(tokens)
    v_types = len(vocab)
    return {
        "L_words": l_words,
        "L_chars_no_spaces": chars_no_spaces,
        "V_types": v_types,
        "V_over_L": (v_types / l_words) if l_words else 0.0,
        "avg_word_len": (sum(word_lengths) / len(word_lengths)) if word_lengths else 0.0,
        "sentence_count": len(sentences),
        "avg_sentence_len_words": (sum(sentence_lengths) / len(sentence_lengths)) if sentence_lengths else 0.0,
        "hapax_count": sum(1 for count in Counter(tokens).values() if count == 1),
    }


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def char_ngrams(text: str, n: int) -> Counter[str]:
    compact = remove_punctuation(clean_text(text)).replace(" ", "_")
    if n <= 0:
        return Counter()
    return Counter(compact[i : i + n] for i in range(0, max(0, len(compact) - n + 1)))


def vocabulary_growth(tokens: list[str], step: int = 100) -> list[dict[str, int]]:
    seen: set[str] = set()
    rows: list[dict[str, int]] = []
    for idx, token in enumerate(tokens, start=1):
        seen.add(token)
        if idx % step == 0 or idx == len(tokens):
            rows.append({"tokens_seen": idx, "vocabulary_size": len(seen)})
    return rows


def zipf_rows(tokens: list[str]) -> list[dict[str, float | int | str]]:
    counts = Counter(tokens)
    rows: list[dict[str, float | int | str]] = []
    for rank, (term, freq) in enumerate(counts.most_common(), start=1):
        rows.append({"rank": rank, "term": term, "frequency": freq, "log_rank": log(rank), "log_frequency": log(freq)})
    return rows

