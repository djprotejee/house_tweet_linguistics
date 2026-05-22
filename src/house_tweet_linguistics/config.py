from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    start_time: str
    end_time: str
    max_posts_per_user: int
    socialdata_max_posts_per_user: int
    socialdata_max_pages_per_account: int
    socialdata_price_per_tweet_usd: float
    min_posts_per_user: int
    exclude_retweets: bool
    exclude_replies: bool
    tweet_text_mode: str
    randomization_iterations: int
    random_seed: int


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or PROJECT_ROOT / "config" / "defaults.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return Settings(**data)


def load_env_file(path: Path | None = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_bearer_token() -> str:
    load_env_file()
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "X_BEARER_TOKEN is missing. Copy .env.example to .env and add your X API bearer token."
        )
    return token


def require_env_var(name: str, help_text: str) -> str:
    load_env_file()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(help_text)
    return value


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
