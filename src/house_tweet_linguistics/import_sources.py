from __future__ import annotations

from html.parser import HTMLParser
from urllib.request import urlopen

import pandas as pd

from .accounts import ACCOUNTS_PATH, REQUIRED_COLUMNS

PRESS_GALLERY_X_HANDLES_URL = "https://pressgallery.house.gov/member-data/members-official-x-handles-119th-congress"
NON_VOTING_JURISDICTIONS = {"AS", "DC", "GU", "MP", "PR", "VI"}


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.cell = ""
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        if tag in {"td", "th"}:
            self.in_cell = True
            self.cell = ""

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell += data

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.in_cell:
            self.row.append(" ".join(self.cell.split()))
            self.in_cell = False
        if tag == "tr" and self.row:
            self.rows.append(self.row)


def import_press_gallery_handles(include_non_voting: bool = False) -> None:
    with urlopen(PRESS_GALLERY_X_HANDLES_URL, timeout=60) as response:
        html = response.read().decode("utf-8", errors="ignore")

    parser = TableParser()
    parser.feed(html)
    source_rows = [row for row in parser.rows if len(row) >= 5 and row[2].startswith("@")]
    if not include_non_voting:
        source_rows = [row for row in source_rows if row[3][:2] not in NON_VOTING_JURISDICTIONS]

    rows = []
    for first_name, last_name, handle, district, party_code, *_ in source_rows:
        if party_code not in {"D", "R", "I"}:
            continue
        party = {"D": "Democratic", "R": "Republican", "I": "Independent"}[party_code]
        rows.append(
            {
                "party": party,
                "name": f"{first_name} {last_name}".strip(),
                "state": district[:2],
                "district": district[2:],
                "twitter_username": handle.lstrip("@"),
                "twitter_id": "",
                "source_url": PRESS_GALLERY_X_HANDLES_URL,
                "included_in_balanced_corpus": "",
                "exclusion_reason": "",
            }
        )

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    ACCOUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ACCOUNTS_PATH, index=False, encoding="utf-8")
    counts = df.groupby("party").size().to_dict()
    print(f"Wrote {len(df)} accounts to {ACCOUNTS_PATH}")
    print(f"Republican: {counts.get('Republican', 0)}")
    print(f"Democratic: {counts.get('Democratic', 0)}")
    print(f"Independent: {counts.get('Independent', 0)}")
