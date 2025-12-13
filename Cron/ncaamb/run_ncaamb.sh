#!/bin/bash
set -e

# Absolute paths only (cron requires this)
PROJECT_DIR="/home/caramey/prod"
VENV_DIR="$PROJECT_DIR/venv"
CRON_DIR="$PROJECT_DIR/Cron/ncaamb"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Run scripts
# python "$CRON_DIR/main.py"
python "$CRON_DIR/send_email.py"