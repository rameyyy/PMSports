#!/bin/bash
set -e

PROJECT_DIR="/home/caramey/prod"
VENV_DIR="$PROJECT_DIR/venv"
CRON_DIR="$PROJECT_DIR/Cron/ufc"
LOG_DIR="/home/caramey/ufc_logs"

mkdir -p "$LOG_DIR"

source "$VENV_DIR/bin/activate"

echo "[$(date)] Running scrape..."
python "$CRON_DIR/scrape.py"

echo "[$(date)] Running predictions..."
python "$CRON_DIR/predict.py"

echo "[$(date)] Running odds update..."
python "$CRON_DIR/oddsapi.py"

echo "[$(date)] Updating homepage stats..."
python "$CRON_DIR/update_homepage.py"
