#!/bin/bash
set -e

# Absolute paths only (cron requires this)
PROJECT_DIR="/home/caramey/prod"
VENV_DIR="$PROJECT_DIR/venv"
CRON_DIR="$PROJECT_DIR/Cron/ufc"

source "$VENV_DIR/bin/activate"

DOW=$(date +%u)  # 1=Mon ... 7=Sun

# Sunday only: scrape Tapology/UFCStats then generate predictions
if [ "$DOW" -eq 7 ]; then
    echo "[$(date)] Running scrape..."
    python "$CRON_DIR/scrape.py"

    echo "[$(date)] Running predictions..."
    python "$CRON_DIR/predict.py"
fi

# Daily: pull odds from The Odds API, update bookmaker tables + prediction_simplified
echo "[$(date)] Running odds update..."
python "$CRON_DIR/oddsapi.py"
