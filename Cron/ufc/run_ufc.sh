#!/bin/bash
set -e

# Absolute paths only (cron requires this)
PROJECT_DIR="/home/caramey/prod"
VENV_DIR="$PROJECT_DIR/venv"
CRON_DIR="$PROJECT_DIR/Cron/ufc"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Odds API — runs daily (8 requests per run, ~500/mo budget)
python "$CRON_DIR/oddsapi.py"

# Full scrape (Tapology + UFCStats) — Sundays only
DOW=$(date +%u)  # 1=Mon ... 7=Sun
if [ "$DOW" -eq 7 ]; then
    python "$CRON_DIR/scrape.py"
fi
