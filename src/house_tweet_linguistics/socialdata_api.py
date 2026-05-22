from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

BASE_URL = "https://api.socialdata.tools/twitter"


class SocialDataClient:
    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}", "Accept": "application/json"})

    def search(self, query: str, cursor: str | None = None, search_type: str = "Latest") -> dict[str, Any]:
        params = f"query={quote(query)}&type={search_type}"
        if cursor:
            params += f"&cursor={quote(cursor)}"
        response = self.session.get(f"{BASE_URL}/search?{params}", timeout=(10, 30))
        if response.status_code == 402:
            raise RuntimeError("SocialData returned 402 Payment Required. Add balance or lower the collection limit.")
        if not response.ok:
            raise RuntimeError(f"SocialData API error {response.status_code}: {response.text}")
        return response.json()
