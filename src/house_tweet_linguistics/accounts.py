from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT

ACCOUNTS_PATH = PROJECT_ROOT / "metadata" / "accounts.csv"
RESOLVED_ACCOUNTS_PATH = PROJECT_ROOT / "metadata" / "accounts_resolved.csv"
INCLUSION_REPORT_PATH = PROJECT_ROOT / "metadata" / "inclusion_report.csv"
MAIN_PARTIES = ("Republican", "Democratic")
PARTIES = ("Republican", "Democratic", "Independent")
PARTY_DIR = {"Republican": "republican", "Democratic": "democratic", "Independent": "independent"}
PARTY_CODE_PREFIX = {"Republican": "rep", "Democratic": "dem", "Independent": "ind"}

REQUIRED_COLUMNS = [
    "party",
    "name",
    "state",
    "district",
    "twitter_username",
    "twitter_id",
    "source_url",
    "included_in_balanced_corpus",
    "exclusion_reason",
]


@dataclass(frozen=True)
class InclusionSet:
    accounts: pd.DataFrame
    report: pd.DataFrame
    n_per_party: int


def load_accounts(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (RESOLVED_ACCOUNTS_PATH if RESOLVED_ACCOUNTS_PATH.exists() else ACCOUNTS_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")
    df["party"] = df["party"].str.strip()
    bad_parties = sorted(set(df.loc[df["party"].ne("") & ~df["party"].isin(PARTIES), "party"]))
    if bad_parties:
        raise ValueError(f"Unsupported party labels: {bad_parties}. Use Republican, Democratic, or Independent.")
    return df


def save_accounts(df: pd.DataFrame, path: Path = RESOLVED_ACCOUNTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def account_tweet_counts(tweets: pd.DataFrame) -> pd.Series:
    if tweets.empty:
        return pd.Series(dtype=int)
    if "twitter_username" in tweets.columns:
        return tweets.assign(
            twitter_username=tweets["twitter_username"].astype(str).str.strip().str.lstrip("@").str.lower()
        ).groupby("twitter_username").size()
    return tweets.groupby("author_id").size()


def make_inclusion_set(
    accounts: pd.DataFrame,
    tweets: pd.DataFrame,
    min_posts: int,
    balanced: bool,
    report_path: Path | None = None,
) -> InclusionSet:
    counts = account_tweet_counts(tweets)
    df = accounts.copy()
    username_key = df["twitter_username"].astype(str).str.strip().str.lstrip("@").str.lower()
    df["tweet_count"] = username_key.map(counts).fillna(0).astype(int)
    df["eligible"] = (
        df["party"].isin(PARTIES)
        & df["twitter_username"].astype(str).str.strip().ne("")
        & df["exclusion_reason"].eq("")
        & df["tweet_count"].ge(min_posts)
    )
    if balanced:
        eligible_counts = df[df["eligible"] & df["party"].isin(MAIN_PARTIES)].groupby("party").size()
        n_per_party = int(eligible_counts.min()) if not eligible_counts.empty and len(eligible_counts) == 2 else 0
    else:
        n_per_party = -1

    included_indices: list[int] = []
    if balanced and n_per_party > 0:
        for party in MAIN_PARTIES:
            party_rows = df[df["eligible"] & df["party"].eq(party)].sort_values(["tweet_count", "name"], ascending=[False, True])
            included_indices.extend(party_rows.head(n_per_party).index.tolist())
        included_indices.extend(df[df["eligible"] & df["party"].eq("Independent")].index.tolist())
    elif not balanced:
        included_indices = df[df["eligible"]].index.tolist()

    df["included_in_balanced_corpus"] = "false"
    df.loc[included_indices, "included_in_balanced_corpus"] = "true"
    df["user_code"] = ""
    for party in PARTIES:
        party_indices = [idx for idx in included_indices if df.loc[idx, "party"] == party]
        for number, idx in enumerate(sorted(party_indices, key=lambda value: df.loc[value, "name"]), start=1):
            df.loc[idx, "user_code"] = f"{PARTY_CODE_PREFIX[party]}{number:03d}"

    report = df[
        [
            "party",
            "name",
            "state",
            "district",
            "twitter_username",
            "twitter_id",
            "tweet_count",
            "eligible",
            "included_in_balanced_corpus",
            "exclusion_reason",
            "user_code",
            "source_url",
        ]
    ].copy()
    output_path = report_path or INCLUSION_REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8")
    return InclusionSet(accounts=df, report=report, n_per_party=n_per_party)
