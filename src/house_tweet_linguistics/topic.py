from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .accounts import MAIN_PARTIES, PARTIES
from .text import clean_text, remove_punctuation, tokenize_words


ISSUE_DICTIONARIES: dict[str, list[str]] = {
    "economy_taxes": [
        "budget",
        "business",
        "businesses",
        "cost",
        "costs",
        "debt",
        "deficit",
        "economy",
        "economic",
        "inflation",
        "job",
        "jobs",
        "manufacturing",
        "price",
        "prices",
        "spending",
        "tariff",
        "tariffs",
        "tax",
        "taxes",
        "wage",
        "wages",
        "worker",
        "workers",
    ],
    "immigration_border": [
        "alien",
        "aliens",
        "asylum",
        "border",
        "borders",
        "cartel",
        "cartels",
        "crossing",
        "crossings",
        "deport",
        "deportation",
        "fentanyl",
        "ice",
        "illegal",
        "immigrant",
        "immigrants",
        "immigration",
        "migrant",
        "migrants",
    ],
    "health_care": [
        "aca",
        "cancer",
        "doctor",
        "doctors",
        "drug",
        "drugs",
        "health",
        "health care",
        "healthcare",
        "hospital",
        "hospitals",
        "medicaid",
        "medicare",
        "mental health",
        "patients",
        "prescription",
        "vaccine",
    ],
    "foreign_policy_ukraine": [
        "ally",
        "allies",
        "china",
        "foreign",
        "gaza",
        "hamas",
        "iran",
        "israel",
        "nato",
        "putin",
        "russia",
        "taiwan",
        "ukraine",
        "ukrainian",
        "war",
    ],
    "energy_climate": [
        "climate",
        "coal",
        "emissions",
        "energy",
        "environment",
        "gas",
        "green",
        "grid",
        "oil",
        "pipeline",
        "renewable",
        "solar",
        "wind",
    ],
    "education": [
        "college",
        "education",
        "school",
        "schools",
        "student",
        "students",
        "teacher",
        "teachers",
        "university",
    ],
    "crime_public_safety": [
        "crime",
        "criminal",
        "criminals",
        "gun",
        "guns",
        "law enforcement",
        "officer",
        "officers",
        "police",
        "safety",
        "violence",
    ],
    "civil_rights": [
        "abortion",
        "civil rights",
        "discrimination",
        "equality",
        "freedom",
        "lgbtq",
        "rights",
        "voting",
        "women",
    ],
    "military_veterans": [
        "air force",
        "army",
        "defense",
        "military",
        "navy",
        "servicemembers",
        "troops",
        "veteran",
        "veterans",
        "va",
    ],
    "government_shutdown": [
        "appropriations",
        "continuing resolution",
        "cr",
        "funding",
        "reopen",
        "shutdown",
        "spending bill",
    ],
    "trump_biden": [
        "biden",
        "bidens",
        "donald",
        "president biden",
        "president trump",
        "trump",
        "trumps",
    ],
    "agriculture_food": [
        "agriculture",
        "crop",
        "crops",
        "dairy",
        "farm",
        "farmer",
        "farmers",
        "farming",
        "food",
        "ranch",
        "ranchers",
        "snap",
    ],
}


def _normalize_username(value: object) -> str:
    return str(value or "").strip().lstrip("@").lower()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)


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


def _party_effects(frame: pd.DataFrame, feature_columns: list[str], label_column: str = "feature") -> pd.DataFrame:
    rows = []
    main = frame[frame["party"].isin(MAIN_PARTIES)].copy()
    for column in feature_columns:
        democratic = main[main["party"].eq("Democratic")][column].astype(float)
        republican = main[main["party"].eq("Republican")][column].astype(float)
        if democratic.dropna().empty or republican.dropna().empty:
            continue
        try:
            _, p_value = mannwhitneyu(republican.dropna(), democratic.dropna(), alternative="two-sided")
        except ValueError:
            p_value = np.nan
        rows.append(
            {
                label_column: column,
                "democratic_mean": float(democratic.mean()),
                "republican_mean": float(republican.mean()),
                "republican_minus_democratic": float(republican.mean() - democratic.mean()),
                "democratic_median": float(democratic.median()),
                "republican_median": float(republican.median()),
                "mann_whitney_u_p_value": float(p_value) if not np.isnan(float(p_value)) else np.nan,
                "cliffs_delta_republican_vs_democratic": _cliffs_delta(republican, democratic),
            }
        )
    return pd.DataFrame(rows).sort_values("mann_whitney_u_p_value", na_position="last")


def _issue_dictionary_rows() -> list[dict[str, str]]:
    rows = []
    for issue, terms in ISSUE_DICTIONARIES.items():
        for term in terms:
            rows.append({"issue": issue, "term": term})
    return rows


def _count_issues(text: str) -> tuple[int, dict[str, int]]:
    normalized = remove_punctuation(clean_text(str(text or ""), keep_hashtags=False))
    tokens = tokenize_words(normalized)
    token_count = len(tokens)
    token_counter = Counter(tokens)
    counts: dict[str, int] = {}
    padded = f" {normalized} "
    for issue, terms in ISSUE_DICTIONARIES.items():
        count = 0
        for term in terms:
            if " " in term:
                phrase = remove_punctuation(clean_text(term, keep_hashtags=False))
                count += len(re.findall(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized))
            else:
                count += token_counter[term]
        counts[issue] = count
    return token_count, counts


def write_issue_dictionary_analysis(tweets: pd.DataFrame, users: pd.DataFrame, tables: Path, figures: Path) -> None:
    pd.DataFrame(_issue_dictionary_rows()).to_csv(tables / "issue_dictionaries.csv", index=False, encoding="utf-8")

    issue_names = list(ISSUE_DICTIONARIES)
    tweet_rows = []
    for _, tweet in tweets.iterrows():
        token_count, counts = _count_issues(str(tweet.get("cleaned_text", "")))
        row = {
            "tweet_id": str(tweet.get("tweet_id", "")),
            "party": tweet.get("party", ""),
            "name": tweet.get("name", ""),
            "twitter_username": tweet.get("twitter_username", ""),
            "twitter_username_key": _normalize_username(tweet.get("twitter_username", "")),
            "created_at": str(tweet.get("created_at", "")),
            "token_count": token_count,
        }
        for issue in issue_names:
            row[f"{issue}_count"] = counts[issue]
            row[f"{issue}_per_1000_words"] = counts[issue] * 1000.0 / token_count if token_count else 0.0
        tweet_rows.append(row)
    by_tweet = pd.DataFrame(tweet_rows)
    by_tweet.to_csv(tables / "issue_scores_by_tweet.csv", index=False, encoding="utf-8")

    user_rows = []
    for _, user in users.iterrows():
        username = _normalize_username(user["twitter_username"])
        subset = by_tweet[by_tweet["twitter_username_key"].eq(username)]
        result = {
            "party": user["party"],
            "name": user["name"],
            "twitter_username": user["twitter_username"],
            "user_code": user["user_code"],
            "tweet_count": int(len(subset)),
            "token_count": int(subset["token_count"].sum()) if not subset.empty else 0,
        }
        for issue in issue_names:
            count = int(subset[f"{issue}_count"].sum()) if not subset.empty else 0
            result[f"{issue}_count"] = count
            result[f"{issue}_per_1000_words"] = count * 1000.0 / result["token_count"] if result["token_count"] else 0.0
        user_rows.append(result)
    by_user = pd.DataFrame(user_rows)
    by_user.to_csv(tables / "issue_scores_by_user.csv", index=False, encoding="utf-8")

    party_rows = []
    for party in PARTIES:
        subset = by_user[by_user["party"].eq(party)]
        if subset.empty:
            continue
        result = {
            "party": party,
            "user_count": int(len(subset)),
            "tweet_count": int(subset["tweet_count"].sum()),
            "token_count": int(subset["token_count"].sum()),
        }
        for issue in issue_names:
            count = int(subset[f"{issue}_count"].sum())
            result[f"{issue}_count"] = count
            result[f"{issue}_per_1000_words"] = count * 1000.0 / result["token_count"] if result["token_count"] else 0.0
        party_rows.append(result)
    by_party = pd.DataFrame(party_rows)
    by_party.to_csv(tables / "issue_scores_by_party.csv", index=False, encoding="utf-8")

    rate_columns = [f"{issue}_per_1000_words" for issue in issue_names]
    effects = _party_effects(by_user, rate_columns, label_column="issue_rate")
    effects.to_csv(tables / "issue_party_effects.csv", index=False, encoding="utf-8")

    if not by_party.empty:
        plot = by_party[by_party["party"].isin(MAIN_PARTIES)].set_index("party")[rate_columns]
        plot.columns = [column.replace("_per_1000_words", "") for column in plot.columns]
        plot = plot.T
        sort_column = "Republican" if "Republican" in plot.columns else plot.columns[0]
        plot = plot.sort_values(sort_column)
        plt.figure(figsize=(10, 7))
        x = np.arange(len(plot.index))
        width = 0.38
        if "Democratic" in plot.columns:
            plt.barh(x - width / 2, plot["Democratic"], width, label="Democratic", color="#2d6cdf")
        if "Republican" in plot.columns:
            plt.barh(x + width / 2, plot["Republican"], width, label="Republican", color="#c43c39")
        plt.yticks(x, plot.index)
        plt.xlabel("Issue markers per 1,000 words")
        plt.title("Issue dictionary rates by party")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "issue_dictionary_rates.png", dpi=160)
        plt.close()

    if not effects.empty:
        top = (
            effects.assign(abs_effect=effects["cliffs_delta_republican_vs_democratic"].abs())
            .sort_values("abs_effect", ascending=False)
            .head(12)
            .sort_values("cliffs_delta_republican_vs_democratic")
        )
        colors = np.where(top["cliffs_delta_republican_vs_democratic"] >= 0, "#c43c39", "#2d6cdf")
        plt.figure(figsize=(8, 5))
        plt.barh(
            top["issue_rate"].str.replace("_per_1000_words", "", regex=False),
            top["cliffs_delta_republican_vs_democratic"],
            color=colors,
        )
        plt.axvline(0, color="#444444", linewidth=0.8)
        plt.xlabel("Cliff's delta: Republican vs Democratic")
        plt.title("Issue dictionary party effects")
        plt.tight_layout()
        plt.savefig(figures / "issue_party_effects.png", dpi=160)
        plt.close()


def write_nmf_topic_model(tweets: pd.DataFrame, users: pd.DataFrame, tables: Path, figures: Path, random_seed: int, n_topics: int = 12) -> None:
    docs = tweets["cleaned_text"].fillna("").astype(str).tolist()
    if len(docs) < n_topics:
        return

    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.65,
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(docs)
    topic_count = min(n_topics, matrix.shape[0], matrix.shape[1])
    model = NMF(n_components=topic_count, init="nndsvda", random_state=random_seed, max_iter=500)
    tweet_topic = _normalize_rows(model.fit_transform(matrix))
    terms = np.array(vectorizer.get_feature_names_out())

    topic_labels = {}
    term_rows = []
    for topic_idx, weights in enumerate(model.components_):
        top_indices = weights.argsort()[::-1][:20]
        top_terms = terms[top_indices]
        topic_id = f"topic_{topic_idx:02d}"
        topic_labels[topic_id] = " / ".join(top_terms[:5])
        for rank, idx in enumerate(top_indices, start=1):
            term_rows.append(
                {
                    "topic_id": topic_id,
                    "topic_label": topic_labels[topic_id],
                    "rank": rank,
                    "term": terms[idx],
                    "weight": float(weights[idx]),
                }
            )
    pd.DataFrame(term_rows).to_csv(tables / "topic_model_terms.csv", index=False, encoding="utf-8")

    topic_cols = [f"topic_{idx:02d}" for idx in range(topic_count)]
    by_tweet = tweets[["tweet_id", "party", "name", "twitter_username", "created_at"]].copy()
    by_tweet["twitter_username_key"] = by_tweet["twitter_username"].map(_normalize_username)
    for idx, column in enumerate(topic_cols):
        by_tweet[column] = tweet_topic[:, idx]
    by_tweet["dominant_topic"] = [topic_cols[idx] for idx in tweet_topic.argmax(axis=1)]
    by_tweet["dominant_topic_label"] = by_tweet["dominant_topic"].map(topic_labels)
    by_tweet.to_csv(tables / "topic_model_by_tweet.csv", index=False, encoding="utf-8")

    user_rows = []
    for _, user in users.iterrows():
        username = _normalize_username(user["twitter_username"])
        subset = by_tweet[by_tweet["twitter_username_key"].eq(username)]
        result = {
            "party": user["party"],
            "name": user["name"],
            "twitter_username": user["twitter_username"],
            "user_code": user["user_code"],
            "tweet_count": int(len(subset)),
        }
        for column in topic_cols:
            result[column] = float(subset[column].mean()) if not subset.empty else 0.0
        if subset.empty:
            result["dominant_topic"] = ""
            result["dominant_topic_label"] = ""
        else:
            dominant = max(topic_cols, key=lambda column: result[column])
            result["dominant_topic"] = dominant
            result["dominant_topic_label"] = topic_labels[dominant]
        user_rows.append(result)
    by_user = pd.DataFrame(user_rows)
    by_user.to_csv(tables / "topic_model_by_user.csv", index=False, encoding="utf-8")

    party_rows = []
    for party in PARTIES:
        subset = by_user[by_user["party"].eq(party)]
        if subset.empty:
            continue
        result = {"party": party, "user_count": int(len(subset)), "tweet_count": int(subset["tweet_count"].sum())}
        for column in topic_cols:
            result[column] = float(subset[column].mean())
        dominant = max(topic_cols, key=lambda column: result[column])
        result["dominant_topic"] = dominant
        result["dominant_topic_label"] = topic_labels[dominant]
        party_rows.append(result)
    by_party = pd.DataFrame(party_rows)
    by_party.to_csv(tables / "topic_model_by_party.csv", index=False, encoding="utf-8")

    effects = _party_effects(by_user, topic_cols, label_column="topic_id")
    effects["topic_label"] = effects["topic_id"].map(topic_labels)
    effects.to_csv(tables / "topic_party_effects.csv", index=False, encoding="utf-8")

    if not by_party.empty:
        plot = by_party[by_party["party"].isin(MAIN_PARTIES)].set_index("party")[topic_cols]
        plot.columns = [f"{column}: {topic_labels[column]}" for column in plot.columns]
        plt.figure(figsize=(11, 4.5))
        plt.imshow(plot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
        plt.yticks(range(len(plot.index)), plot.index)
        plt.xticks(range(len(plot.columns)), plot.columns, rotation=45, ha="right", fontsize=8)
        plt.colorbar(label="Mean topic share")
        plt.title("NMF topic shares by party")
        plt.tight_layout()
        plt.savefig(figures / "topic_party_heatmap.png", dpi=170)
        plt.close()

    if not effects.empty:
        plot = effects.copy()
        plot["label"] = plot["topic_id"] + ": " + plot["topic_label"]
        plot = plot.sort_values("republican_minus_democratic")
        colors = np.where(plot["republican_minus_democratic"] >= 0, "#c43c39", "#2d6cdf")
        plt.figure(figsize=(10, 6))
        plt.barh(plot["label"], plot["republican_minus_democratic"], color=colors)
        plt.axvline(0, color="#444444", linewidth=0.8)
        plt.xlabel("Mean topic share difference: Republican minus Democratic")
        plt.title("NMF topic differences by party")
        plt.tight_layout()
        plt.savefig(figures / "topic_party_differences.png", dpi=170)
        plt.close()


def write_party_classifier(users: pd.DataFrame, tables: Path, figures: Path, random_seed: int) -> None:
    main = users[users["party"].isin(MAIN_PARTIES)].copy()
    if main["party"].nunique() < 2:
        return
    labels = main["party"].astype(str).to_numpy()
    texts = main["text"].fillna("").astype(str).to_numpy()
    min_class = int(pd.Series(labels).value_counts().min())
    folds = max(2, min(5, min_class))

    model = make_pipeline(
        TfidfVectorizer(min_df=2, max_df=0.95, max_features=12000, ngram_range=(1, 2), stop_words="english"),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=random_seed),
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    predictions = cross_val_predict(model, texts, labels, cv=cv)
    baseline = float(pd.Series(labels).value_counts(normalize=True).max())
    summary = pd.DataFrame(
        [
            {
                "level": "account",
                "feature_source": "full_account_tfidf",
                "folds": folds,
                "n_accounts": int(len(main)),
                "baseline_majority_accuracy": baseline,
                "accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
                "macro_f1": float(f1_score(labels, predictions, average="macro")),
            }
        ]
    )
    summary.to_csv(tables / "party_classifier_summary.csv", index=False, encoding="utf-8")

    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"}).to_csv(
        tables / "party_classifier_classification_report.csv", index=False, encoding="utf-8"
    )

    prediction_rows = main[["party", "name", "twitter_username", "user_code"]].copy()
    prediction_rows["predicted_party"] = predictions
    prediction_rows["correct"] = prediction_rows["party"].eq(prediction_rows["predicted_party"])
    prediction_rows.to_csv(tables / "party_classifier_predictions.csv", index=False, encoding="utf-8")

    labels_order = ["Democratic", "Republican"]
    cm = confusion_matrix(labels, predictions, labels=labels_order)
    pd.DataFrame(cm, index=labels_order, columns=labels_order).to_csv(tables / "party_classifier_confusion_matrix.csv", encoding="utf-8")
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels_order)), labels_order, rotation=20, ha="right")
    plt.yticks(range(len(labels_order)), labels_order)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Party classifier confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111111")
    plt.tight_layout()
    plt.savefig(figures / "party_classifier_confusion_matrix.png", dpi=170)
    plt.close()

    model.fit(texts, labels)
    vectorizer = model.named_steps["tfidfvectorizer"]
    classifier = model.named_steps["logisticregression"]
    features = np.array(vectorizer.get_feature_names_out())
    classes = list(classifier.classes_)
    coefficients = classifier.coef_[0]
    positive_class = classes[1]
    rows = []
    for idx in coefficients.argsort()[::-1][:60]:
        rows.append({"party_signal": positive_class, "term": features[idx], "coefficient": float(coefficients[idx])})
    for idx in coefficients.argsort()[:60]:
        rows.append({"party_signal": classes[0], "term": features[idx], "coefficient": float(coefficients[idx])})
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(tables / "party_classifier_top_features.csv", index=False, encoding="utf-8")

    plot_df = pd.concat(
        [
            coef_df[coef_df["party_signal"].eq(classes[0])].head(15),
            coef_df[coef_df["party_signal"].eq(positive_class)].head(15),
        ],
        ignore_index=True,
    ).sort_values("coefficient")
    colors = np.where(plot_df["coefficient"] >= 0, "#c43c39", "#2d6cdf")
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["term"], plot_df["coefficient"], color=colors)
    plt.axvline(0, color="#444444", linewidth=0.8)
    plt.xlabel("Logistic regression coefficient")
    plt.title("Strongest party-classifier terms")
    plt.tight_layout()
    plt.savefig(figures / "party_classifier_top_features.png", dpi=170)
    plt.close()


def write_topic_issue_and_classifier_analysis(
    tweets: pd.DataFrame,
    users: pd.DataFrame,
    tables: Path,
    figures: Path,
    random_seed: int,
) -> None:
    write_nmf_topic_model(tweets, users, tables, figures, random_seed=random_seed)
    write_issue_dictionary_analysis(tweets, users, tables, figures)
    write_party_classifier(users, tables, figures, random_seed=random_seed)
