#!/bin/bash
# NOTE: intentionally NOT using `set -e`. Each stage is run independently so a
# transient failure in one (e.g. a scrape blip or a brief DB network drop) does
# not skip the remaining stages for the whole day.

# Only run during NCAAMB season (Nov 1 - Apr 15)
MONTH=$(date +%-m)
DAY=$(date +%-d)
if [ "$MONTH" -ge 5 ] && [ "$MONTH" -le 10 ]; then
  exit 0
fi
if [ "$MONTH" -eq 4 ] && [ "$DAY" -gt 15 ]; then
  exit 0
fi

# Absolute paths only (cron requires this)
PROJECT_DIR="/home/caramey/prod"
VENV_DIR="$PROJECT_DIR/venv"
CRON_DIR="$PROJECT_DIR/Cron/ncaamb"
LOG_DIR="/home/caramey/ncaamb_logs"

mkdir -p "$LOG_DIR"

source "$VENV_DIR/bin/activate"

FAILED=()

# run_stage <label> <script-relative-to-CRON_DIR> <logfile>
run_stage() {
    local label="$1" script="$2" logfile="$3"
    echo "[$(date)] Running $label..."
    if ! python "$CRON_DIR/$script" >> "$logfile" 2>&1; then
        echo "[$(date)] ✗ $label FAILED (see $logfile)"
        FAILED+=("$label")
    fi
}

DATE_TAG=$(date +%Y-%m-%d)

run_stage "main"              main.py                              "$LOG_DIR/main_${DATE_TAG}.log"
run_stage "pick of day"       pick_of_day.py                       "$LOG_DIR/pick_of_day_${DATE_TAG}.log"
run_stage "homepage update"   ui_jobs/homepage_update.py           "$LOG_DIR/homepage_${DATE_TAG}.log"
run_stage "ui games update"   ui_jobs/update_ui_games.py           "$LOG_DIR/ui_games_${DATE_TAG}.log"
run_stage "model performance" update_model_performance.py          "$LOG_DIR/model_perf_${DATE_TAG}.log"
run_stage "bracket results"   marchmadness/update_bracket_results.py "$LOG_DIR/bracket_${DATE_TAG}.log"

if [ ${#FAILED[@]} -ne 0 ]; then
    echo "[$(date)] Pipeline completed with failures: ${FAILED[*]}"
    exit 1
fi
echo "[$(date)] Pipeline completed OK"
