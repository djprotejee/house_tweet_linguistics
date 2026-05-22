$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
.\.venv\Scripts\python.exe -m house_tweet_linguistics collect --resolve-users
.\.venv\Scripts\python.exe -m house_tweet_linguistics collect --fetch-tweets
.\.venv\Scripts\python.exe -m house_tweet_linguistics mirrors --balanced
.\.venv\Scripts\python.exe -m house_tweet_linguistics analyze --balanced

