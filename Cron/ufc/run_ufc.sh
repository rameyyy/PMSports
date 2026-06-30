#!/bin/bash
# NOTE: intentionally NOT using `set -e`. Each stage is run independently so a
# transient failure in one (e.g. a scrape blip or a brief DB network drop) does
# not skip the remaining stages (predict / odds / homepage) for the whole day.

# Absolute paths only (cron requires this)
PROJECT_DIR="/home/caramey/prod"
VENV_DIR="$PROJECT_DIR/venv"
CRON_DIR="$PROJECT_DIR/Cron/ufc"
LOG_DIR="/home/caramey/ufc_logs"

mkdir -p "$LOG_DIR"

source "$VENV_DIR/bin/activate"

DOW=$(date +%u)  # 1=Mon ... 7=Sun
FAILED=()

# run_stage <label> <script> <logfile>
run_stage() {
    local label="$1" script="$2" logfile="$3"
    echo "[$(date)] Running $label..."
    if ! python "$CRON_DIR/$script" >> "$logfile" 2>&1; then
        echo "[$(date)] ✗ $label FAILED (see $logfile)"
        FAILED+=("$label")
    fi
}

# Sunday only: scrape Tapology/UFCStats then generate predictions
if [ "$DOW" -eq 7 ]; then
    run_stage "scrape"      scrape.py  "$LOG_DIR/scrape_$(date +%Y-%m-%d).log"
    run_stage "predictions" predict.py "$LOG_DIR/predict_$(date +%Y-%m-%d).log"
fi

# Daily: pull odds from The Odds API, update bookmaker tables + prediction_simplified
run_stage "odds update" oddsapi.py "$LOG_DIR/odds_$(date +%Y-%m-%d).log"

# Sunday only: update homepage stats table (runs after odds so EV calcs have fresh odds)
if [ "$DOW" -eq 7 ]; then
    run_stage "homepage stats" update_homepage.py "$LOG_DIR/homepage_$(date +%Y-%m-%d).log"
fi

if [ ${#FAILED[@]} -ne 0 ]; then
    echo "[$(date)] Pipeline completed with failures: ${FAILED[*]}"
    exit 1
fi
echo "[$(date)] Pipeline completed OK"
