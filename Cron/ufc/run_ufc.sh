#!/bin/bash
set -e

# Absolute paths only (cron requires this)
PROJECT_DIR="/home/caramey/prod"
VENV_DIR="$PROJECT_DIR/venv"
CRON_DIR="$PROJECT_DIR/Cron/ufc"
LOG_DIR="/home/caramey/ufc_logs"

mkdir -p "$LOG_DIR"

source "$VENV_DIR/bin/activate"

DOW=$(date +%u)  # 1=Mon ... 7=Sun

# Sunday only: scrape Tapology/UFCStats then generate predictions
if [ "$DOW" -eq 7 ]; then
    echo "[$(date)] Running scrape..."
    python "$CRON_DIR/scrape.py" >> "$LOG_DIR/scrape_$(date +%Y-%m-%d).log" 2>&1

    echo "[$(date)] Running predictions..."
    python "$CRON_DIR/predict.py" >> "$LOG_DIR/predict_$(date +%Y-%m-%d).log" 2>&1
fi

# Daily: pull odds from The Odds API, update bookmaker tables + prediction_simplified
echo "[$(date)] Running odds update..."
python "$CRON_DIR/oddsapi.py" >> "$LOG_DIR/odds_$(date +%Y-%m-%d).log" 2>&1

# Sunday only: update homepage stats table (runs after odds so EV calcs have fresh odds)
if [ "$DOW" -eq 7 ]; then
    echo "[$(date)] Updating homepage stats..."
    python "$CRON_DIR/update_homepage.py" >> "$LOG_DIR/homepage_$(date +%Y-%m-%d).log" 2>&1
fi
