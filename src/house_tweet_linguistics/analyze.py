from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from .accounts import MAIN_PARTIES, PARTIES, PARTY_DIR, make_inclusion_set, load_accounts
from .collect import corpus_tweets_path, tweets_dataframe
from .config import PROJECT_ROOT, ensure_dirs, load_settings
from .style import STYLE_NUMERIC_COLUMNS, style_metrics, summarize_style_effects
from .text import char_ngrams, clean_text, ngrams, remove_punctuation, text_metrics, tokenize_words, vocabulary_growth, zipf_rows
from .topic import write_topic_issue_and_classifier_analysis


TABLES = PROJECT_ROOT / "data_tables"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"


def _included_data(balanced: bool, corpus: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = load_settings()
    accounts = load_accounts()
    tweets = tweets_dataframe(corpus)
    if tweets.empty:
        raise RuntimeError(f"No tweets found in {corpus_tweets_path(corpus)}. Run collection first.")
    mode_name = "balanced" if balanced else "full"
    inclusion_report = PROJECT_ROOT / "metadata" / f"inclusion_report_{corpus}_{mode_name}.csv"
    inclusion = make_inclusion_set(accounts, tweets, settings.min_posts_per_user, balanced=balanced, report_path=inclusion_report)
    included = inclusion.accounts[inclusion.accounts["included_in_balanced_corpus"].eq("true")].copy()
    if included.empty:
        raise RuntimeError(f"No included accounts. Check min_posts_per_user and {inclusion_report}.")
    usernames = set(included["twitter_username"].astype(str).str.strip().str.lstrip("@").str.lower())
    selected_tweets = tweets[
        tweets["twitter_username"].astype(str).str.strip().str.lstrip("@").str.lower().isin(usernames)
    ].copy()
    return included, selected_tweets


def _plot_zipf(df: pd.DataFrame, label: str, output: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.loglog(df["rank"], df["frequency"], marker=".", linestyle="none")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"Zipf plot - {label}")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _plot_heaps(df: pd.DataFrame, label: str, output: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.loglog(df["tokens_seen"], df["vocabulary_size"], marker=".", linestyle="-")
    plt.xlabel("Tokens seen")
    plt.ylabel("Vocabulary size")
    plt.title(f"Heaps plot - {label}")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _fit_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(slope), float(intercept)


def _combined_texts_by_user(accounts: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, account in accounts.iterrows():
        username = str(account["twitter_username"]).strip().lstrip("@").lower()
        user_tweets = tweets[
            tweets["twitter_username"].astype(str).str.strip().str.lstrip("@").str.lower().eq(username)
        ].sort_values("created_at")
        text = "\n".join(user_tweets["cleaned_text"].fillna("").astype(str).tolist())
        rows.append(
            {
                "author_id": str(user_tweets["author_id"].iloc[0]) if not user_tweets.empty else "",
                "party": account["party"],
                "name": account["name"],
                "twitter_username": account["twitter_username"],
                "user_code": account["user_code"],
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def _feature_rows(level: str, rows: list[dict[str, str]]) -> pd.DataFrame:
    output_rows = []
    for row in rows:
        metrics = text_metrics(row["text"])
        metrics.update({key: value for key, value in row.items() if key != "text"})
        metrics["level"] = level
        output_rows.append(metrics)
    return pd.DataFrame(output_rows)


def write_basic_features(accounts: pd.DataFrame, tweets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tweet_rows = []
    for _, tweet in tweets.iterrows():
        row = {
            "tweet_id": str(tweet["tweet_id"]),
            "author_id": str(tweet["author_id"]),
            "party": tweet["party"],
            "name": tweet["name"],
            "twitter_username": tweet["twitter_username"],
            "created_at": tweet["created_at"],
            "text": str(tweet["cleaned_text"]),
        }
        tweet_rows.append(row)
    by_tweet = _feature_rows("tweet", tweet_rows)

    users = _combined_texts_by_user(accounts, tweets)
    by_user = _feature_rows("user", users.to_dict("records"))

    party_rows = []
    for party in PARTIES:
        party_text = "\n".join(users[users["party"].eq(party)]["text"].tolist())
        if not party_text.strip():
            continue
        party_rows.append({"party": party, "party_dir": PARTY_DIR[party], "text": party_text})
    by_party = _feature_rows("party", party_rows)

    by_tweet.to_csv(TABLES / "features_by_tweet.csv", index=False, encoding="utf-8")
    by_user.to_csv(TABLES / "features_by_user.csv", index=False, encoding="utf-8")
    by_party.to_csv(TABLES / "features_by_party.csv", index=False, encoding="utf-8")
    return by_tweet, by_user, by_party


def write_zipf_heaps(users: pd.DataFrame) -> None:
    all_zipf = []
    all_heaps = []
    fits = []
    for party in PARTIES:
        text = "\n".join(users[users["party"].eq(party)]["text"].tolist())
        tokens = tokenize_words(remove_punctuation(clean_text(text)))
        if not tokens:
            continue
        zipf = pd.DataFrame(zipf_rows(tokens))
        if not zipf.empty:
            zipf["party"] = party
            all_zipf.append(zipf)
            slope, intercept = _fit_loglog(zipf["rank"].to_numpy(dtype=float), zipf["frequency"].to_numpy(dtype=float))
            fits.append({"law": "Zipf", "party": party, "slope": slope, "intercept": intercept})
            _plot_zipf(zipf, party, FIGURES / f"zipf_{party.lower()}.png")
        heaps = pd.DataFrame(vocabulary_growth(tokens, step=max(50, len(tokens) // 200 if tokens else 50)))
        if not heaps.empty:
            heaps["party"] = party
            all_heaps.append(heaps)
            slope, intercept = _fit_loglog(heaps["tokens_seen"].to_numpy(dtype=float), heaps["vocabulary_size"].to_numpy(dtype=float))
            fits.append({"law": "Heaps", "party": party, "slope": slope, "intercept": intercept})
            _plot_heaps(heaps, party, FIGURES / f"heaps_{party.lower()}.png")
    if all_zipf:
        pd.concat(all_zipf, ignore_index=True).to_csv(TABLES / "zipf_by_party.csv", index=False, encoding="utf-8")
    if all_heaps:
        pd.concat(all_heaps, ignore_index=True).to_csv(TABLES / "heaps_by_party.csv", index=False, encoding="utf-8")
    pd.DataFrame(fits).to_csv(TABLES / "law_fits.csv", index=False, encoding="utf-8")


def write_ngrams(users: pd.DataFrame) -> None:
    rows = []
    for party in PARTIES:
        text = "\n".join(users[users["party"].eq(party)]["text"].tolist())
        tokens = tokenize_words(remove_punctuation(clean_text(text)))
        if not tokens:
            continue
        for n in [1, 2, 3]:
            for gram, freq in ngrams(tokens, n).most_common(200):
                rows.append({"party": party, "type": "word", "n": n, "ngram": " ".join(gram), "frequency": freq})
        for n in [3, 4, 5]:
            for gram, freq in char_ngrams(text, n).most_common(200):
                rows.append({"party": party, "type": "char", "n": n, "ngram": gram, "frequency": freq})
    pd.DataFrame(rows).to_csv(TABLES / "ngrams_by_party.csv", index=False, encoding="utf-8")


def write_keywords(users: pd.DataFrame) -> None:
    party_texts = {
        party: "\n".join(users[users["party"].eq(party)]["text"].tolist())
        for party in MAIN_PARTIES
    }
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
    tfidf = vectorizer.fit_transform([party_texts["Republican"], party_texts["Democratic"]])
    terms = np.array(vectorizer.get_feature_names_out())
    tfidf_rows = []
    for party_index, party in enumerate(MAIN_PARTIES):
        scores = np.asarray(tfidf[party_index].todense()).ravel()
        for idx in scores.argsort()[::-1][:200]:
            tfidf_rows.append({"party": party, "term": terms[idx], "tfidf": float(scores[idx])})

    counts = {
        party: Counter(tokenize_words(remove_punctuation(clean_text(text))))
        for party, text in party_texts.items()
    }
    vocab = sorted(set(counts["Republican"]) | set(counts["Democratic"]))
    alpha = 0.5
    rows = []
    for party, other in [("Republican", "Democratic"), ("Democratic", "Republican")]:
        total = sum(counts[party].values())
        other_total = sum(counts[other].values())
        for term in vocab:
            party_prob = (counts[party][term] + alpha) / (total + alpha * len(vocab))
            other_prob = (counts[other][term] + alpha) / (other_total + alpha * len(vocab))
            rows.append(
                {
                    "party": party,
                    "term": term,
                    "frequency": counts[party][term],
                    "other_frequency": counts[other][term],
                    "smoothed_log_ratio": float(np.log(party_prob / other_prob)),
                }
            )
    keyword_df = pd.DataFrame(rows)
    tfidf_df = pd.DataFrame(tfidf_rows)
    merged = keyword_df.merge(tfidf_df, on=["party", "term"], how="left").fillna({"tfidf": 0.0})
    merged.sort_values(["party", "smoothed_log_ratio"], ascending=[True, False]).to_csv(
        TABLES / "key_terms_by_party.csv", index=False, encoding="utf-8"
    )


def write_distances_and_clusters(users: pd.DataFrame, settings_seed: int) -> None:
    if len(users) < 3:
        raise RuntimeError("At least 3 included accounts are required for distances and clustering.")
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(users["text"].fillna("").astype(str))
    cosine = cosine_distances(matrix)

    count_vectorizer = CountVectorizer(min_df=1, stop_words="english")
    count_matrix = count_vectorizer.fit_transform(users["text"].fillna("").astype(str)).toarray().astype(float)
    count_matrix = count_matrix + 1e-12
    count_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)

    pair_rows = []
    for i, j in combinations(range(len(users)), 2):
        party_i = users.iloc[i]["party"]
        party_j = users.iloc[j]["party"]
        if party_i == party_j == "Republican":
            relation = "R-R"
        elif party_i == party_j == "Democratic":
            relation = "D-D"
        elif party_i == party_j == "Independent":
            relation = "I-I"
        elif "Independent" in {party_i, party_j}:
            relation = "I-other"
        else:
            relation = "R-D"
        pair_rows.append(
            {
                "user_i": users.iloc[i]["user_code"],
                "user_j": users.iloc[j]["user_code"],
                "name_i": users.iloc[i]["name"],
                "name_j": users.iloc[j]["name"],
                "party_i": party_i,
                "party_j": party_j,
                "relation": relation,
                "cosine_distance": float(cosine[i, j]),
                "jensen_shannon_distance": float(jensenshannon(count_matrix[i], count_matrix[j])),
            }
        )
    distances = pd.DataFrame(pair_rows)
    distances.to_csv(TABLES / "distances_by_user.csv", index=False, encoding="utf-8")
    distances.groupby("relation")[["cosine_distance", "jensen_shannon_distance"]].agg(["mean", "median", "std", "count"]).to_csv(
        REPORTS / "distance_summary.csv", encoding="utf-8"
    )

    try:
        clustering = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
    except TypeError:
        clustering = AgglomerativeClustering(n_clusters=2, affinity="cosine", linkage="average")
    labels = clustering.fit_predict(matrix.toarray())
    main_mask = users["party"].isin(MAIN_PARTIES).to_numpy()
    party_binary = users.loc[main_mask, "party"].map({"Republican": 0, "Democratic": 1}).to_numpy()
    ari = adjusted_rand_score(party_binary, labels[main_mask]) if len(set(party_binary)) == 2 else 0.0
    sil = silhouette_score(cosine, labels, metric="precomputed") if len(set(labels)) > 1 else 0.0

    if matrix.shape[1] >= 2 and matrix.shape[0] >= 2:
        svd = TruncatedSVD(n_components=2, random_state=settings_seed)
        coords = svd.fit_transform(matrix)
    else:
        coords = np.zeros((len(users), 2))

    cluster_df = users[["author_id", "party", "name", "twitter_username", "user_code"]].copy()
    cluster_df["cluster"] = labels
    cluster_df["x"] = coords[:, 0]
    cluster_df["y"] = coords[:, 1]
    cluster_df["adjusted_rand_index"] = ari
    cluster_df["silhouette_cosine"] = sil
    cluster_df.to_csv(TABLES / "clustering_by_user.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(8, 6))
    colors = users["party"].map({"Republican": "#c43c39", "Democratic": "#2d6cdf", "Independent": "#4c956c"}).tolist()
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=28, alpha=0.85)
    for idx, row in users.iterrows():
        plt.annotate(row["user_code"], (coords[idx, 0], coords[idx, 1]), fontsize=7, alpha=0.75)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("2D TF-IDF map by account")
    plt.tight_layout()
    plt.savefig(FIGURES / "accounts_2d_map.png", dpi=160)
    plt.close()

    summary = distances.groupby("relation")["cosine_distance"].mean()
    plt.figure(figsize=(6, 4))
    summary.reindex(["R-R", "D-D", "R-D", "I-other", "I-I"]).dropna().plot(
        kind="bar", color=["#c43c39", "#2d6cdf", "#626262", "#4c956c", "#88b04b"]
    )
    plt.ylabel("Mean cosine distance")
    plt.title("Mean account distances")
    plt.tight_layout()
    plt.savefig(FIGURES / "distance_relations.png", dpi=160)
    plt.close()


def write_randomization(users: pd.DataFrame) -> None:
    settings = load_settings()
    distances = pd.read_csv(TABLES / "distances_by_user.csv")
    observed_between = distances[distances["relation"].eq("R-D")]["cosine_distance"].mean()
    observed_within = distances[distances["relation"].isin(["R-R", "D-D"])]["cosine_distance"].mean()
    observed_gap = observed_between - observed_within
    rng = np.random.default_rng(settings.random_seed)
    users = users[users["party"].isin(MAIN_PARTIES)].reset_index(drop=True)
    if len(users) < 3:
        return
    labels = users["party"].to_numpy()
    user_codes = users["user_code"].to_numpy()
    pair_distance = {
        tuple(sorted((row["user_i"], row["user_j"]))): row["cosine_distance"]
        for _, row in distances.iterrows()
    }
    gaps = []
    for _ in range(settings.randomization_iterations):
        shuffled = rng.permutation(labels)
        label_by_user = dict(zip(user_codes, shuffled))
        between = []
        within = []
        for (left, right), distance in pair_distance.items():
            if left not in label_by_user or right not in label_by_user:
                continue
            if label_by_user[left] == label_by_user[right]:
                within.append(distance)
            else:
                between.append(distance)
        gaps.append(float(np.mean(between) - np.mean(within)))
    p_value = (sum(gap >= observed_gap for gap in gaps) + 1) / (len(gaps) + 1)
    pd.DataFrame(
        [
            {
                "observed_between_mean": observed_between,
                "observed_within_mean": observed_within,
                "observed_gap": observed_gap,
                "iterations": settings.randomization_iterations,
                "one_sided_p_value": p_value,
                "random_gap_mean": float(np.mean(gaps)),
                "random_gap_std": float(np.std(gaps)),
            }
        ]
    ).to_csv(REPORTS / "randomization_report.csv", index=False, encoding="utf-8")


def write_networks(users: pd.DataFrame) -> None:
    rows = []
    for party in PARTIES:
        text = "\n".join(users[users["party"].eq(party)]["text"].tolist())
        tokens = tokenize_words(remove_punctuation(clean_text(text)))
        if not tokens:
            continue
        edges = Counter(zip(tokens, tokens[1:]))
        graph = nx.Graph()
        for (left, right), weight in edges.items():
            if left != right:
                graph.add_edge(left, right, weight=weight)
        rows.append(
            {
                "party": party,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
                "average_clustering": nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0.0,
                "connected_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "network_features_by_party.csv", index=False, encoding="utf-8")


def _repetition_metrics(text: str) -> dict[str, float | int]:
    tokens = tokenize_words(remove_punctuation(clean_text(text)))
    counts = Counter(tokens)
    total = sum(counts.values())
    repeated_tokens = sum(freq for freq in counts.values() if freq > 1)
    top_freq = max(counts.values()) if counts else 0
    entropy = 0.0
    if total:
        probs = np.array([freq / total for freq in counts.values()], dtype=float)
        entropy = float(-(probs * np.log2(probs)).sum())
    return {
        "token_count": total,
        "type_count": len(counts),
        "hapax_count": sum(1 for freq in counts.values() if freq == 1),
        "repeated_token_count": repeated_tokens,
        "repeated_token_share": (repeated_tokens / total) if total else 0.0,
        "top_token_share": (top_freq / total) if total else 0.0,
        "lexical_entropy": entropy,
    }


def write_repetition(users: pd.DataFrame) -> None:
    user_rows = []
    for _, row in users.iterrows():
        metrics = _repetition_metrics(str(row["text"]))
        metrics.update(
            {
                "party": row["party"],
                "name": row["name"],
                "twitter_username": row["twitter_username"],
                "user_code": row["user_code"],
            }
        )
        user_rows.append(metrics)
    pd.DataFrame(user_rows).to_csv(TABLES / "repetition_by_user.csv", index=False, encoding="utf-8")

    party_rows = []
    for party in PARTIES:
        text = "\n".join(users[users["party"].eq(party)]["text"].tolist())
        if not text.strip():
            continue
        metrics = _repetition_metrics(text)
        metrics.update({"party": party})
        party_rows.append(metrics)
    pd.DataFrame(party_rows).to_csv(TABLES / "repetition_by_party.csv", index=False, encoding="utf-8")


def write_style_features(tweets: pd.DataFrame, users: pd.DataFrame) -> None:
    tweet_rows = []
    for _, tweet in tweets.iterrows():
        row = style_metrics(
            raw_text=str(tweet.get("raw_text", "")),
            cleaned_text=str(tweet.get("cleaned_text", "")),
        )
        username = str(tweet.get("twitter_username", "")).strip().lstrip("@")
        row.update(
            {
                "tweet_id": str(tweet.get("tweet_id", "")),
                "author_id": str(tweet.get("author_id", "")),
                "party": tweet.get("party", ""),
                "name": tweet.get("name", ""),
                "twitter_username": username,
                "twitter_username_key": username.lower(),
                "created_at": str(tweet.get("created_at", "")),
            }
        )
        tweet_rows.append(row)

    by_tweet = pd.DataFrame(tweet_rows)
    by_tweet.to_csv(TABLES / "style_features_by_tweet.csv", index=False, encoding="utf-8")
    if by_tweet.empty:
        return

    numeric_cols = [column for column in STYLE_NUMERIC_COLUMNS if column in by_tweet.columns]
    user_rows = []
    for _, user in users.iterrows():
        username = str(user["twitter_username"]).strip().lstrip("@")
        subset = by_tweet[by_tweet["twitter_username_key"].eq(username.lower())]
        result = {
            "party": user["party"],
            "name": user["name"],
            "twitter_username": user["twitter_username"],
            "user_code": user["user_code"],
            "tweet_count": int(len(subset)),
        }
        for column in numeric_cols:
            result[column] = float(subset[column].mean()) if not subset.empty else np.nan
        user_rows.append(result)
    by_user = pd.DataFrame(user_rows)
    by_user.to_csv(TABLES / "style_features_by_user.csv", index=False, encoding="utf-8")

    party_rows = []
    for party, subset in by_tweet.groupby("party", sort=True):
        result = {"party": party, "tweet_count": int(len(subset))}
        for column in numeric_cols:
            result[column] = float(subset[column].mean())
        party_rows.append(result)
    pd.DataFrame(party_rows).to_csv(TABLES / "style_features_by_party.csv", index=False, encoding="utf-8")

    effects = summarize_style_effects(by_user)
    effects.to_csv(TABLES / "style_party_effects.csv", index=False, encoding="utf-8")

    numeric_user = by_user[numeric_cols].copy()
    if len(numeric_user.columns) >= 2:
        corr = numeric_user.corr(method="spearman").reset_index().rename(columns={"index": "feature"})
        corr.to_csv(TABLES / "style_feature_correlations_by_user.csv", index=False, encoding="utf-8")

    if not effects.empty:
        effect_col = "cliffs_delta_republican_vs_democratic"
        plot_rows = (
            effects.assign(abs_effect=effects[effect_col].abs())
            .sort_values("abs_effect", ascending=False)
            .head(16)
            .sort_values(effect_col)
        )
        colors = np.where(plot_rows[effect_col] >= 0, "#c43c39", "#2d6cdf")
        plt.figure(figsize=(8, 6))
        plt.barh(plot_rows["feature"], plot_rows[effect_col], color=colors)
        plt.axvline(0, color="#444444", linewidth=0.8)
        plt.xlabel("Cliff's delta: Republican vs Democratic")
        plt.title("Largest stylistic party effects")
        plt.tight_layout()
        plt.savefig(FIGURES / "style_party_effects.png", dpi=160)
        plt.close()


def write_correlations_and_autocorrelations(users: pd.DataFrame, tweets: pd.DataFrame) -> None:
    features_path = TABLES / "features_by_user.csv"
    if features_path.exists():
        features = pd.read_csv(features_path)
        numeric = features.select_dtypes(include=[np.number]).copy()
        if len(numeric.columns) >= 2:
            corr = numeric.corr(method="spearman").reset_index().rename(columns={"index": "feature"})
            corr.to_csv(TABLES / "feature_correlations_by_user.csv", index=False, encoding="utf-8")

        main = features[features["party"].isin(MAIN_PARTIES)].copy()
        if not main.empty:
            main["party_binary_republican"] = main["party"].map({"Democratic": 0, "Republican": 1}).astype(float)
            rows = []
            for column in numeric.columns:
                if column not in main.columns:
                    continue
                democratic = main[main["party"].eq("Democratic")][column].astype(float)
                republican = main[main["party"].eq("Republican")][column].astype(float)
                values = main[column].astype(float)
                rows.append(
                    {
                        "feature": column,
                        "democratic_mean": float(democratic.mean()) if len(democratic) else 0.0,
                        "republican_mean": float(republican.mean()) if len(republican) else 0.0,
                        "republican_minus_democratic": float(republican.mean() - democratic.mean()) if len(democratic) and len(republican) else 0.0,
                        "spearman_with_republican_label": float(values.corr(main["party_binary_republican"], method="spearman")),
                    }
                )
            pd.DataFrame(rows).to_csv(TABLES / "party_feature_effects.csv", index=False, encoding="utf-8")

    rows = []
    metrics = ["L_words", "L_chars_no_spaces", "V_types", "V_over_L", "avg_word_len", "avg_sentence_len_words"]
    tweet_features = []
    for _, tweet in tweets.iterrows():
        row = text_metrics(str(tweet.get("cleaned_text", "")))
        row.update(
            {
                "tweet_id": str(tweet.get("tweet_id", "")),
                "twitter_username": str(tweet.get("twitter_username", "")).strip().lstrip("@").lower(),
                "created_at": str(tweet.get("created_at", "")),
            }
        )
        tweet_features.append(row)
    tweet_df = pd.DataFrame(tweet_features)
    for _, user in users.iterrows():
        username = str(user["twitter_username"]).strip().lstrip("@").lower()
        user_tweets = tweet_df[tweet_df["twitter_username"].eq(username)].sort_values("created_at")
        result = {
            "party": user["party"],
            "name": user["name"],
            "twitter_username": user["twitter_username"],
            "user_code": user["user_code"],
            "tweet_count": len(user_tweets),
        }
        for metric in metrics:
            series = user_tweets[metric].astype(float).to_numpy() if metric in user_tweets else np.array([])
            if len(series) >= 3 and np.std(series[:-1]) > 0 and np.std(series[1:]) > 0:
                result[f"{metric}_lag1_autocorrelation"] = float(np.corrcoef(series[:-1], series[1:])[0, 1])
            else:
                result[f"{metric}_lag1_autocorrelation"] = np.nan
        rows.append(result)
    pd.DataFrame(rows).to_csv(TABLES / "temporal_autocorrelation_by_user.csv", index=False, encoding="utf-8")


def run_analysis(balanced: bool = True, corpus: str = "strict") -> None:
    global TABLES, REPORTS, FIGURES
    settings = load_settings()
    mode_name = "balanced" if balanced else "full"
    output_name = f"{corpus}_{mode_name}"
    TABLES = PROJECT_ROOT / "data_tables" / output_name
    REPORTS = PROJECT_ROOT / "reports" / output_name
    FIGURES = REPORTS / "figures"
    ensure_dirs([TABLES, REPORTS, FIGURES])
    accounts, tweets = _included_data(balanced, corpus)
    write_basic_features(accounts, tweets)
    users = _combined_texts_by_user(accounts, tweets)
    write_zipf_heaps(users)
    write_ngrams(users)
    write_keywords(users)
    write_distances_and_clusters(users.reset_index(drop=True), settings.random_seed)
    write_randomization(users.reset_index(drop=True))
    write_networks(users)
    write_repetition(users)
    write_style_features(tweets, users)
    write_topic_issue_and_classifier_analysis(tweets, users, TABLES, FIGURES, settings.random_seed)
    write_correlations_and_autocorrelations(users, tweets)
    print(f"Analysis complete. Corpus={corpus}. Balanced={balanced}. Tables: {TABLES}. Figures: {FIGURES}.")
