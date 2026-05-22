from __future__ import annotations

import argparse

from .analyze import run_analysis
from .audit import audit_accounts
from .collect import fetch_tweets, fetch_tweets_socialdata, rebuild_tweets_from_socialdata_raw, resolve_users, socialdata_cost_estimate
from .import_sources import import_press_gallery_handles
from .mirrors import build_text_mirrors


def main() -> None:
    parser = argparse.ArgumentParser(prog="house-tweet-linguistics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Resolve X usernames and/or fetch tweets.")
    collect_parser.add_argument("--resolve-users", action="store_true", help="Resolve twitter_username values to X user IDs.")
    collect_parser.add_argument("--fetch-tweets", action="store_true", help="Fetch posts from X user timelines.")
    collect_parser.add_argument("--fetch-socialdata", action="store_true", help="Fetch posts using SocialData search API.")
    collect_parser.add_argument("--estimate-socialdata-cost", action="store_true", help="Estimate SocialData upper-bound tweet cost.")
    collect_parser.add_argument("--rebuild-from-socialdata-raw", action="store_true", help="Rebuild clean tweets.jsonl from saved raw SocialData pages without API calls.")
    collect_parser.add_argument("--test-socialdata", metavar="USERNAME", help="Fetch a tiny SocialData sample for one username.")
    collect_parser.add_argument("--test-limit", type=int, default=5, help="Maximum posts for --test-socialdata.")
    collect_parser.add_argument("--max-accounts", type=int, default=None, help="Safety cap for SocialData accounts in one run.")
    collect_parser.add_argument("--max-pages", type=int, default=None, help="Safety cap for SocialData result pages in one run.")
    collect_parser.add_argument("--max-pages-per-account", type=int, default=None, help="Safety cap for SocialData pages per account.")
    collect_parser.add_argument("--budget-usd", type=float, default=None, help="Required safety budget cap for --fetch-socialdata.")
    collect_parser.add_argument("--only-usernames", nargs="+", default=None, help="Limit SocialData collection to these usernames.")
    collect_parser.add_argument("--ignore-socialdata-skip", action="store_true", help="Retry usernames even if they are in the SocialData skip list.")

    import_parser = subparsers.add_parser("import-handles", help="Import official House Press Gallery X handles.")
    import_parser.add_argument(
        "--include-non-voting",
        action="store_true",
        help="Include non-voting delegates and resident commissioner entries.",
    )

    subparsers.add_parser("audit-accounts", help="Validate account metadata completeness and duplicates.")

    mirrors_parser = subparsers.add_parser("mirrors", help="Build UTF-8 text mirrors.")
    mirrors_parser.add_argument("--balanced", action="store_true", help="Use balanced Republican/Democratic account set.")
    mirrors_parser.add_argument("--full", action="store_true", help="Use all eligible accounts.")
    mirrors_parser.add_argument("--text-mode", choices=["cleaned", "raw"], default=None)
    mirrors_parser.add_argument("--corpus", choices=["strict", "extended"], default="strict", help="Tweet corpus variant to mirror.")

    analyze_parser = subparsers.add_parser("analyze", help="Run statistical-linguistic analysis.")
    analyze_parser.add_argument("--balanced", action="store_true", help="Use balanced Republican/Democratic account set.")
    analyze_parser.add_argument("--full", action="store_true", help="Use all eligible accounts.")
    analyze_parser.add_argument("--corpus", choices=["strict", "extended"], default="strict", help="Tweet corpus variant to analyze.")

    all_parser = subparsers.add_parser("all", help="Fetch tweets, build mirrors, and analyze.")
    all_parser.add_argument("--skip-resolve", action="store_true", help="Do not resolve usernames before fetching.")
    all_parser.add_argument("--full", action="store_true", help="Use all eligible accounts instead of balanced corpus.")

    args = parser.parse_args()

    if args.command == "collect":
        if not args.resolve_users and not args.fetch_tweets and not args.fetch_socialdata and not args.estimate_socialdata_cost and not args.test_socialdata and not args.rebuild_from_socialdata_raw:
            parser.error("collect requires --resolve-users, --fetch-tweets, --fetch-socialdata, --test-socialdata, --rebuild-from-socialdata-raw, and/or --estimate-socialdata-cost")
        if args.estimate_socialdata_cost:
            socialdata_cost_estimate()
        if args.rebuild_from_socialdata_raw:
            rebuild_tweets_from_socialdata_raw()
        if args.resolve_users:
            resolve_users()
        if args.fetch_tweets:
            fetch_tweets()
        if args.fetch_socialdata:
            if args.budget_usd is None:
                parser.error("--fetch-socialdata requires --budget-usd so collection cannot spend without an explicit cap")
            fetch_tweets_socialdata(
                max_accounts=args.max_accounts,
                max_pages=args.max_pages,
                max_pages_per_account=args.max_pages_per_account,
                budget_usd=args.budget_usd,
                only_usernames=args.only_usernames,
                ignore_skip=args.ignore_socialdata_skip,
            )
        if args.test_socialdata:
            fetch_tweets_socialdata(
                test_username=args.test_socialdata,
                test_limit=args.test_limit,
                max_accounts=1,
                max_pages=args.max_pages,
                max_pages_per_account=args.max_pages_per_account,
                budget_usd=args.budget_usd,
            )
    elif args.command == "import-handles":
        import_press_gallery_handles(include_non_voting=args.include_non_voting)
    elif args.command == "audit-accounts":
        audit_accounts()
    elif args.command == "mirrors":
        build_text_mirrors(balanced=not args.full, text_mode=args.text_mode, corpus=args.corpus)
    elif args.command == "analyze":
        run_analysis(balanced=not args.full, corpus=args.corpus)
    elif args.command == "all":
        if not args.skip_resolve:
            resolve_users()
        fetch_tweets()
        build_text_mirrors(balanced=not args.full)
        run_analysis(balanced=not args.full)
