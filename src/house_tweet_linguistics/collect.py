from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import pandas as pd

from .accounts import load_accounts, save_accounts
from .config import PROJECT_ROOT, append_jsonl, load_settings, read_jsonl, require_bearer_token, require_env_var, write_jsonl
from .socialdata_api import SocialDataClient
from .text import clean_text, remove_punctuation
from .x_api import XApiClient

TWEETS_PATH = PROJECT_ROOT / "data_json" / "tweets.jsonl"
TWEETS_EXTENDED_PATH = PROJECT_ROOT / "data_json" / "tweets_extended_window.jsonl"
SOCIALDATA_RAW_PAGES_PATH = PROJECT_ROOT / "data_json" / "socialdata_raw_pages.jsonl"
COLLECTION_ERRORS_PATH = PROJECT_ROOT / "metadata" / "collection_errors.csv"
SOCIALDATA_SKIP_PATH = PROJECT_ROOT / "metadata" / "socialdata_skip_accounts.csv"


def _safe_print(message: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def resolve_users() -> None:
    accounts = load_accounts()
    token = require_bearer_token()
    client = XApiClient(token)
    usernames = accounts["twitter_username"].dropna().astype(str).str.strip().str.lstrip("@")
    resolved = client.lookup_usernames(usernames)
    updated = accounts.copy()
    for idx, row in updated.iterrows():
        username = str(row["twitter_username"]).strip().lstrip("@").lower()
        if not username:
            continue
        user = resolved.get(username)
        if user:
            updated.loc[idx, "twitter_id"] = user["id"]
    save_accounts(updated)
    print(f"Resolved {len(resolved)} usernames. Wrote metadata/accounts_resolved.csv")


def _existing_tweet_ids(path: Path) -> set[str]:
    return {str(row.get("tweet_id", "")) for row in read_jsonl(path)}


def _existing_socialdata_counts(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    counts: dict[str, int] = {}
    for row in rows:
        username = str(row.get("twitter_username", "")).strip().lstrip("@").lower()
        if not username:
            continue
        counts[username] = counts.get(username, 0) + 1
    return counts


def _raw_page_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("twitter_username", "")).lower(),
        str(row.get("cursor_in", "")),
        str(row.get("query", "")),
    )


def _existing_raw_page_keys(path: Path) -> set[tuple[str, str, str]]:
    return {_raw_page_key(row) for row in read_jsonl(path)}


def _load_socialdata_skip_usernames() -> set[str]:
    if not SOCIALDATA_SKIP_PATH.exists():
        return set()
    usernames: set[str] = set()
    for line in SOCIALDATA_SKIP_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split(",", 2)
        if parts and parts[0] and parts[0] != "twitter_username":
            usernames.add(parts[0].strip().lower())
    return usernames


def _append_socialdata_skip(username: str, reason: str) -> None:
    SOCIALDATA_SKIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SOCIALDATA_SKIP_PATH.exists():
        SOCIALDATA_SKIP_PATH.write_text("twitter_username,reason\n", encoding="utf-8")
    with SOCIALDATA_SKIP_PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{username.lower()},{reason}\n")


def _tweet_row_from_socialdata(account: pd.Series, tweet: dict, start_dt: datetime, end_dt: datetime) -> dict | None:
    tweet_id = str(tweet.get("id_str", ""))
    if not tweet_id:
        return None
    if tweet.get("retweeted_status"):
        return None
    created_at = tweet.get("tweet_created_at", "")
    if created_at:
        created_dt = _iso_to_datetime(created_at)
        if created_dt < start_dt or created_dt >= end_dt:
            return None
    raw_text = tweet.get("full_text") or tweet.get("text") or ""
    if not raw_text:
        return None
    cleaned = clean_text(raw_text)
    return {
        "tweet_id": tweet_id,
        "author_id": str(tweet.get("user", {}).get("id_str", "")),
        "party": account["party"],
        "name": account["name"],
        "state": account["state"],
        "district": account["district"],
        "twitter_username": account["twitter_username"],
        "created_at": created_at,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "cleaned_no_punct": remove_punctuation(cleaned),
        "metrics": {
            "reply_count": tweet.get("reply_count"),
            "retweet_count": tweet.get("retweet_count"),
            "quote_count": tweet.get("quote_count"),
            "favorite_count": tweet.get("favorite_count"),
            "views_count": tweet.get("views_count"),
        },
        "flags": {
            "lang": tweet.get("lang", ""),
            "is_quote_status": tweet.get("is_quote_status", False),
            "quoted_status_id_str": tweet.get("quoted_status_id_str"),
            "source_provider": "socialdata",
        },
    }


def fetch_tweets() -> None:
    settings = load_settings()
    accounts = load_accounts()
    eligible = accounts[
        accounts["party"].isin(["Republican", "Democratic", "Independent"])
        & accounts["twitter_id"].astype(str).str.strip().ne("")
        & accounts["exclusion_reason"].astype(str).str.strip().eq("")
    ].copy()
    if eligible.empty:
        raise RuntimeError("No eligible accounts with twitter_id. Fill metadata/accounts.csv and run collect --resolve-users.")

    token = require_bearer_token()
    client = XApiClient(token)
    existing_ids = _existing_tweet_ids(TWEETS_PATH)
    new_rows: list[dict] = []
    for _, account in eligible.iterrows():
        tweets = client.user_tweets(
            user_id=str(account["twitter_id"]),
            start_time=settings.start_time,
            end_time=settings.end_time,
            max_posts=settings.max_posts_per_user,
            exclude_retweets=settings.exclude_retweets,
            exclude_replies=settings.exclude_replies,
        )
        for tweet in tweets:
            tweet_id = str(tweet["id"])
            if tweet_id in existing_ids:
                continue
            raw_text = tweet.get("text", "")
            cleaned = clean_text(raw_text)
            row = {
                "tweet_id": tweet_id,
                "author_id": str(account["twitter_id"]),
                "party": account["party"],
                "name": account["name"],
                "state": account["state"],
                "district": account["district"],
                "twitter_username": account["twitter_username"],
                "created_at": tweet.get("created_at", ""),
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "cleaned_no_punct": remove_punctuation(cleaned),
                "metrics": tweet.get("public_metrics", {}),
                "flags": {
                    "lang": tweet.get("lang", ""),
                    "possibly_sensitive": tweet.get("possibly_sensitive", False),
                    "referenced_tweets": tweet.get("referenced_tweets", []),
                },
            }
            new_rows.append(row)
            existing_ids.add(tweet_id)
        _safe_print(f"{account['party']}: {account['name']} @{account['twitter_username']} - {len(tweets)} posts")
        if new_rows:
            append_jsonl(TWEETS_PATH, new_rows)
            new_rows.clear()

    all_rows = read_jsonl(TWEETS_PATH)
    all_rows = sorted(
        all_rows,
        key=lambda item: (item.get("party", ""), item.get("name", ""), item.get("created_at", ""), item.get("tweet_id", "")),
    )
    write_jsonl(TWEETS_PATH, all_rows)
    print(f"Wrote {len(all_rows)} total tweet records to {TWEETS_PATH}")


def _iso_to_unix_seconds(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())


def _iso_to_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def socialdata_cost_estimate() -> None:
    settings = load_settings()
    accounts = load_accounts()
    eligible = accounts[
        accounts["party"].isin(["Republican", "Democratic", "Independent"])
        & accounts["twitter_username"].astype(str).str.strip().ne("")
        & accounts["exclusion_reason"].astype(str).str.strip().eq("")
    ].copy()
    tweets_per_page = 20
    pages_per_account = (settings.socialdata_max_posts_per_user + tweets_per_page - 1) // tweets_per_page
    estimated_returned_tweets = len(eligible) * pages_per_account * tweets_per_page
    saved_tweets = len(eligible) * settings.socialdata_max_posts_per_user
    estimated = estimated_returned_tweets * settings.socialdata_price_per_tweet_usd
    print(f"Eligible accounts: {len(eligible)}")
    print(f"Max posts per account: {settings.socialdata_max_posts_per_user}")
    print(f"Target saved tweets: {saved_tweets}")
    print(f"Estimated returned tweets charged: {estimated_returned_tweets}")
    print(f"Estimated upper-bound tweet cost: ${estimated:.2f}")


def fetch_tweets_socialdata(
    test_username: str | None = None,
    test_limit: int | None = None,
    max_accounts: int | None = None,
    max_pages: int | None = None,
    max_pages_per_account: int | None = None,
    budget_usd: float | None = None,
    only_usernames: list[str] | None = None,
    ignore_skip: bool = False,
) -> None:
    settings = load_settings()
    accounts = load_accounts()
    eligible = accounts[
        accounts["party"].isin(["Republican", "Democratic", "Independent"])
        & accounts["twitter_username"].astype(str).str.strip().ne("")
        & accounts["exclusion_reason"].astype(str).str.strip().eq("")
    ].copy()
    if eligible.empty:
        raise RuntimeError("No eligible accounts with twitter_username. Run import-handles first.")

    api_key = require_env_var(
        "SOCIALDATA_API_KEY",
        "SOCIALDATA_API_KEY is missing. Add it to .env before using the SocialData collector.",
    )
    client = SocialDataClient(api_key)
    existing_ids = _existing_tweet_ids(TWEETS_PATH)
    existing_counts = _existing_socialdata_counts(TWEETS_PATH)
    existing_raw_page_keys = _existing_raw_page_keys(SOCIALDATA_RAW_PAGES_PATH)
    skip_usernames = _load_socialdata_skip_usernames()
    start_ts = _iso_to_unix_seconds(settings.start_time)
    end_ts = _iso_to_unix_seconds(settings.end_time)
    start_dt = _iso_to_datetime(settings.start_time)
    end_dt = _iso_to_datetime(settings.end_time)
    run_pages = 0
    run_returned_tweets = 0
    run_cost = 0.0

    if test_username:
        needle = test_username.strip().lstrip("@").lower()
        eligible = eligible[eligible["twitter_username"].astype(str).str.strip().str.lstrip("@").str.lower().eq(needle)]
        if eligible.empty:
            raise RuntimeError(f"Username not found in metadata/accounts.csv: {test_username}")
    if only_usernames:
        needles = {username.strip().lstrip("@").lower() for username in only_usernames if username.strip()}
        eligible = eligible[
            eligible["twitter_username"]
            .astype(str)
            .str.strip()
            .str.lstrip("@")
            .str.lower()
            .isin(needles)
        ]
        if eligible.empty:
            raise RuntimeError(f"No usernames from --only-usernames found in metadata/accounts.csv: {', '.join(sorted(needles))}")

    max_posts_per_user = test_limit or settings.socialdata_max_posts_per_user
    account_page_cap = max_pages_per_account or settings.socialdata_max_pages_per_account
    if max_accounts is not None:
        eligible = eligible.head(max_accounts)

    for _, account in eligible.iterrows():
        username = str(account["twitter_username"]).strip().lstrip("@")
        if not ignore_skip and username.lower() in skip_usernames:
            _safe_print(f"{account['party']}: {account['name']} @{username} - skipped by prior no-growth/empty marker")
            continue
        already_saved = existing_counts.get(username.lower(), 0)
        if already_saved >= max_posts_per_user:
            _safe_print(f"{account['party']}: {account['name']} @{username} - already has {already_saved} posts, skipping")
            continue
        query = f"from:{username} -filter:replies -filter:retweets since_time:{start_ts} until_time:{end_ts}"
        collected = already_saved
        cursor = None
        account_pages = 0
        account_saved_this_run = 0
        account_returned_this_run = 0
        while collected < max_posts_per_user:
            if account_pages >= account_page_cap:
                _safe_print(
                    f"{account['party']}: {account['name']} @{username} - "
                    f"account page cap reached: {account_pages}/{account_page_cap}, total {collected}"
                )
                break
            if max_pages is not None and run_pages >= max_pages:
                _safe_print(f"Run page cap reached: {run_pages}/{max_pages}. Stopping safely.")
                break
            if budget_usd is not None and run_cost >= budget_usd:
                _safe_print(f"Run budget cap reached: ${run_cost:.4f}/${budget_usd:.4f}. Stopping safely.")
                break
            payload = None
            last_exc: Exception | None = None
            for attempt in range(1, 4):
                try:
                    payload = client.search(query=query, cursor=cursor, search_type="Latest")
                    break
                except Exception as exc:
                    last_exc = exc
                    _safe_print(f"{account['party']}: {account['name']} @{username} - request attempt {attempt}/3 failed: {type(exc).__name__}")
                    time.sleep(2 * attempt)
            if payload is None:
                assert last_exc is not None
                COLLECTION_ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with COLLECTION_ERRORS_PATH.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(f"{account['party']},{account['name']},{username},{type(last_exc).__name__},{str(last_exc).replace(',', ';')}\n")
                _safe_print(f"{account['party']}: {account['name']} @{username} - stopped at {collected} posts because {type(last_exc).__name__}")
                break
            tweets = payload.get("tweets", [])
            run_pages += 1
            account_pages += 1
            run_returned_tweets += len(tweets)
            account_returned_this_run += len(tweets)
            run_cost = run_returned_tweets * settings.socialdata_price_per_tweet_usd
            raw_page = {
                "party": account["party"],
                "name": account["name"],
                "state": account["state"],
                "district": account["district"],
                "twitter_username": account["twitter_username"],
                "query": query,
                "cursor_in": cursor or "",
                "next_cursor": payload.get("next_cursor") or "",
                "returned_count": len(tweets),
                "tweets": tweets,
            }
            raw_page_key = _raw_page_key(raw_page)
            if raw_page_key not in existing_raw_page_keys:
                append_jsonl(SOCIALDATA_RAW_PAGES_PATH, [raw_page])
                existing_raw_page_keys.add(raw_page_key)
            if not tweets:
                break
            before_page = collected
            page_rows: list[dict] = []
            for tweet in tweets:
                tweet_row = _tweet_row_from_socialdata(account, tweet, start_dt, end_dt)
                if tweet_row is None or tweet_row["tweet_id"] in existing_ids:
                    continue
                page_rows.append(tweet_row)
                existing_ids.add(tweet_row["tweet_id"])
                collected += 1
                if collected >= max_posts_per_user:
                    break
            if page_rows:
                append_jsonl(TWEETS_PATH, page_rows)
                account_saved_this_run += len(page_rows)
            cursor = payload.get("next_cursor")
            _safe_print(
                f"{account['party']}: {account['name']} @{username} - "
                f"page saved {collected - before_page}, total {collected}, "
                f"account pages {account_pages}/{account_page_cap}, "
                f"run pages {run_pages}, returned {run_returned_tweets}, est ${run_cost:.4f}"
            )
            if not cursor:
                break
        _safe_print(f"{account['party']}: {account['name']} @{username} - {collected} posts")
        if collected < max_posts_per_user and account_pages >= account_page_cap and account_saved_this_run == 0:
            _append_socialdata_skip(username, "page_cap_no_growth")
            skip_usernames.add(username.lower())
        elif collected == 0 and account_returned_this_run == 0:
            _append_socialdata_skip(username, "empty_result")
            skip_usernames.add(username.lower())
        if test_username:
            break
        if max_pages is not None and run_pages >= max_pages:
            break
        if budget_usd is not None and run_cost >= budget_usd:
            break

    all_rows = read_jsonl(TWEETS_PATH)
    all_rows = sorted(
        all_rows,
        key=lambda item: (item.get("party", ""), item.get("name", ""), item.get("created_at", ""), item.get("tweet_id", "")),
    )
    write_jsonl(TWEETS_PATH, all_rows)
    print(f"Wrote {len(all_rows)} total tweet records to {TWEETS_PATH}")
    print(f"Run estimate: {run_pages} pages, {run_returned_tweets} returned tweets, ${run_cost:.4f}")


def rebuild_tweets_from_socialdata_raw() -> None:
    settings = load_settings()
    start_dt = _iso_to_datetime(settings.start_time)
    end_dt = _iso_to_datetime(settings.end_time)
    pages = read_jsonl(SOCIALDATA_RAW_PAGES_PATH)
    existing = {row["tweet_id"]: row for row in read_jsonl(TWEETS_PATH)}
    added = 0
    for page in pages:
        account = pd.Series(
            {
                "party": page.get("party", ""),
                "name": page.get("name", ""),
                "state": page.get("state", ""),
                "district": page.get("district", ""),
                "twitter_username": page.get("twitter_username", ""),
            }
        )
        for tweet in page.get("tweets", []):
            row = _tweet_row_from_socialdata(account, tweet, start_dt, end_dt)
            if row is None or row["tweet_id"] in existing:
                continue
            existing[row["tweet_id"]] = row
            added += 1
    rows = sorted(
        existing.values(),
        key=lambda item: (item.get("party", ""), item.get("name", ""), item.get("created_at", ""), item.get("tweet_id", "")),
    )
    write_jsonl(TWEETS_PATH, list(rows))
    print(f"Rebuilt {TWEETS_PATH} from raw SocialData pages. Added {added}; total {len(rows)}.")


def corpus_tweets_path(corpus: str = "strict") -> Path:
    if corpus == "strict":
        return TWEETS_PATH
    if corpus == "extended":
        return TWEETS_EXTENDED_PATH
    raise ValueError(f"Unsupported corpus: {corpus}. Use strict or extended.")


def tweets_dataframe(corpus: str = "strict") -> pd.DataFrame:
    rows = read_jsonl(corpus_tweets_path(corpus))
    return pd.DataFrame(rows)
