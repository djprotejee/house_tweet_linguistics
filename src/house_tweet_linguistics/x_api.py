from __future__ import annotations

import time
from typing import Any, Iterable

import requests

BASE_URL = "https://api.x.com/2"


class XApiClient:
    def __init__(self, bearer_token: str, sleep_on_rate_limit: bool = True) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})
        self.sleep_on_rate_limit = sleep_on_rate_limit

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        while True:
            response = self.session.get(url, params=params, timeout=60)
            if response.status_code == 429 and self.sleep_on_rate_limit:
                reset = response.headers.get("x-rate-limit-reset")
                retry_after = response.headers.get("retry-after")
                if reset and reset.isdigit():
                    wait_seconds = max(5, int(reset) - int(time.time()) + 5)
                elif retry_after and retry_after.isdigit():
                    wait_seconds = int(retry_after) + 5
                else:
                    wait_seconds = 60
                time.sleep(wait_seconds)
                continue
            if not response.ok:
                raise RuntimeError(f"X API error {response.status_code}: {response.text}")
            return response.json()

    def lookup_usernames(self, usernames: Iterable[str]) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        cleaned = [username.strip().lstrip("@") for username in usernames if username and username.strip()]
        for start in range(0, len(cleaned), 100):
            batch = cleaned[start : start + 100]
            payload = self.get(
                "/users/by",
                params={
                    "usernames": ",".join(batch),
                    "user.fields": "id,name,username,verified,verified_type,url,description",
                },
            )
            for user in payload.get("data", []):
                resolved[user["username"].lower()] = user
        return resolved

    def user_tweets(
        self,
        user_id: str,
        start_time: str,
        end_time: str,
        max_posts: int,
        exclude_retweets: bool = True,
        exclude_replies: bool = True,
    ) -> list[dict[str, Any]]:
        tweets: list[dict[str, Any]] = []
        pagination_token: str | None = None
        exclude = []
        if exclude_retweets:
            exclude.append("retweets")
        if exclude_replies:
            exclude.append("replies")

        while len(tweets) < max_posts:
            params: dict[str, Any] = {
                "max_results": 100,
                "start_time": start_time,
                "end_time": end_time,
                "tweet.fields": "id,text,created_at,public_metrics,lang,possibly_sensitive,referenced_tweets",
            }
            if exclude:
                params["exclude"] = ",".join(exclude)
            if pagination_token:
                params["pagination_token"] = pagination_token
            payload = self.get(f"/users/{user_id}/tweets", params=params)
            rows = payload.get("data", [])
            tweets.extend(rows[: max_posts - len(tweets)])
            meta = payload.get("meta", {})
            pagination_token = meta.get("next_token")
            if not pagination_token or not rows:
                break
        return tweets

