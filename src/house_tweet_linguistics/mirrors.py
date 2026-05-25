from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .accounts import PARTY_DIR, make_inclusion_set, load_accounts
from .collect import corpus_tweets_path, tweets_dataframe
from .config import PROJECT_ROOT, ensure_dirs, load_settings


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_text_mirrors(balanced: bool = True, text_mode: str | None = None, corpus: str = "strict") -> None:
    settings = load_settings()
    mode = text_mode or settings.tweet_text_mode
    accounts = load_accounts()
    tweets = tweets_dataframe(corpus)
    if tweets.empty:
        raise RuntimeError(f"No tweets found in {corpus_tweets_path(corpus)}. Run collection first.")
    mode_name = "balanced" if balanced else "full"
    inclusion_report = PROJECT_ROOT / "metadata" / f"inclusion_report_{corpus}_{mode_name}.csv"
    inclusion = make_inclusion_set(accounts, tweets, settings.min_posts_per_user, balanced=balanced, report_path=inclusion_report)
    included = inclusion.accounts[inclusion.accounts["included_in_balanced_corpus"].eq("true")]
    if included.empty:
        raise RuntimeError(f"No accounts satisfy inclusion rules. Check {inclusion_report} after this run.")

    base = PROJECT_ROOT / "data_txt" / corpus / mode_name
    _reset_dir(base / "tweet_level")
    _reset_dir(base / "user_level")
    _reset_dir(base / "party_level")

    text_column = "cleaned_text" if mode == "cleaned" else "raw_text"
    party_texts: dict[str, list[str]] = {"Republican": [], "Democratic": [], "Independent": []}
    account_rows = []
    tweet_rows = []

    for _, account in included.iterrows():
        party = account["party"]
        party_dir = PARTY_DIR[party]
        user_code = account["user_code"]
        username = str(account["twitter_username"]).strip().lstrip("@").lower()
        user_tweets = tweets[
            tweets["twitter_username"].astype(str).str.strip().str.lstrip("@").str.lower().eq(username)
        ].sort_values("created_at")
        account_rows.append(
            {
                "corpus": corpus,
                "mode": mode_name,
                "party": party,
                "party_dir": party_dir,
                "user_code": user_code,
                "name": account["name"],
                "state": account["state"],
                "district": account["district"],
                "twitter_username": account["twitter_username"],
                "twitter_id": account.get("twitter_id", ""),
                "tweet_count": int(account["tweet_count"]),
                "source_url": account.get("source_url", ""),
            }
        )
        tweet_dir = base / "tweet_level" / party_dir / user_code
        user_dir = base / "user_level" / party_dir
        ensure_dirs([tweet_dir, user_dir])
        user_texts: list[str] = []
        for number, (_, tweet) in enumerate(user_tweets.iterrows(), start=1):
            text = str(tweet.get(text_column, "")).strip()
            if not text:
                continue
            user_texts.append(text)
            tweet_file = tweet_dir / f"tweet_{number:06d}.txt"
            tweet_file.write_text(text + "\n", encoding="utf-8", newline="\n")
            tweet_rows.append(
                {
                    "corpus": corpus,
                    "mode": mode_name,
                    "party": party,
                    "party_dir": party_dir,
                    "user_code": user_code,
                    "name": account["name"],
                    "state": account["state"],
                    "district": account["district"],
                    "twitter_username": account["twitter_username"],
                    "tweet_number": number,
                    "tweet_file": str(tweet_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "tweet_id": str(tweet.get("tweet_id", "")),
                    "author_id": str(tweet.get("author_id", "")),
                    "created_at": str(tweet.get("created_at", "")),
                    "text_mode": mode,
                }
            )
        merged = "\n".join(user_texts).strip()
        (user_dir / f"{user_code}.txt").write_text(merged + "\n", encoding="utf-8", newline="\n")
        party_texts[party].append(merged)

    for party, texts in party_texts.items():
        party_dir = PARTY_DIR[party]
        (base / "party_level" / f"{party_dir}.txt").write_text("\n".join(texts).strip() + "\n", encoding="utf-8", newline="\n")

    pd.DataFrame(account_rows).sort_values(["party_dir", "user_code"]).to_csv(
        base / "account_codebook.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(tweet_rows).sort_values(["party_dir", "user_code", "tweet_number"]).to_csv(
        base / "tweet_manifest.csv", index=False, encoding="utf-8"
    )

    print(f"Built text mirrors in {base}. Corpus={corpus}. Balanced={balanced}. N per party={inclusion.n_per_party}")
