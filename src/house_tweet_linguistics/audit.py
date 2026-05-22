from __future__ import annotations

import pandas as pd

from .accounts import MAIN_PARTIES, PARTIES, load_accounts
from .config import PROJECT_ROOT


AUDIT_PATH = PROJECT_ROOT / "metadata" / "account_audit.csv"


def audit_accounts() -> None:
    accounts = load_accounts()
    rows = []
    for idx, row in accounts.iterrows():
        issues = []
        if row["party"] not in PARTIES:
            issues.append("invalid_party")
        if not str(row["name"]).strip():
            issues.append("missing_name")
        if not str(row["twitter_username"]).strip():
            issues.append("missing_twitter_username")
        if not str(row["source_url"]).strip():
            issues.append("missing_source_url")
        if str(row["exclusion_reason"]).strip() and str(row["included_in_balanced_corpus"]).lower() == "true":
            issues.append("included_with_exclusion_reason")
        rows.append(
            {
                "row_number": idx + 2,
                "party": row["party"],
                "name": row["name"],
                "twitter_username": row["twitter_username"],
                "source_url": row["source_url"],
                "issues": ";".join(issues),
            }
        )

    audit = pd.DataFrame(rows)
    if not accounts.empty:
        duplicate_usernames = accounts["twitter_username"].str.lower().str.strip()
        duplicates = duplicate_usernames[duplicate_usernames.ne("") & duplicate_usernames.duplicated(keep=False)]
        if not duplicates.empty:
            duplicate_rows = accounts[duplicate_usernames.isin(set(duplicates))]
            extra = pd.DataFrame(
                [
                    {
                        "row_number": idx + 2,
                        "party": row["party"],
                        "name": row["name"],
                        "twitter_username": row["twitter_username"],
                        "source_url": row["source_url"],
                        "issues": "duplicate_twitter_username",
                    }
                    for idx, row in duplicate_rows.iterrows()
                ]
            )
            audit = pd.concat([audit, extra], ignore_index=True)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8")

    counts = accounts[accounts["party"].isin(PARTIES)].groupby("party").size().reindex(PARTIES, fill_value=0)
    complete = accounts[
        accounts["party"].isin(PARTIES)
        & accounts["twitter_username"].astype(str).str.strip().ne("")
        & accounts["source_url"].astype(str).str.strip().ne("")
    ].groupby("party").size().reindex(PARTIES, fill_value=0)
    print("Account audit written to metadata/account_audit.csv")
    print("Rows by party:")
    for party in PARTIES:
        print(f"  {party}: {counts[party]} rows, {complete[party]} with username and source_url")
    print(f"Main R/D rows: {int(counts.reindex(MAIN_PARTIES, fill_value=0).sum())}")
