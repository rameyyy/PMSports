#!/usr/bin/env python3
"""
Create the bracket_predictions table in the ncaamb database.

Run from ncaamb/ directory:
    python marchmadness/create_bracket_table.py

Table design:
  - bracket_slot  : fixed bracket position (e.g. 'East_R64_G1') — the anchor
                    that links your predicted matchup to the actual game regardless
                    of which teams end up playing.
  - pred_team_*   : who YOU predicted would play in that slot (from your model)
  - prob_*        : model probabilities P(team_1 wins)
  - predicted_winner : your pick
  - game_id       : FK to games table — filled in by update_bracket_results.py
                    after the actual game is scraped
  - actual_team_* : who actually played (may differ from pred_team_* if your
                    earlier picks were wrong)
  - actual_winner : filled from games table after game is played
  - correct       : NULL=not played yet, 1=your pick was right, 0=wrong
"""

import sys
from pathlib import Path

ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

from scrapes.sqlconn import create_connection

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bracket_predictions (
    id               INT            NOT NULL AUTO_INCREMENT,
    bracket_year     SMALLINT       NOT NULL DEFAULT 2026,

    -- Fixed bracket position — the anchor for matching predictions to reality
    bracket_slot     VARCHAR(30)    NOT NULL,

    -- Tournament structure
    round            VARCHAR(30)    NOT NULL,
    region           VARCHAR(20)    DEFAULT NULL,

    -- Your predicted matchup (what the model said would happen)
    pred_team_1      VARCHAR(250)   NOT NULL,
    pred_team_1_seed TINYINT        DEFAULT NULL,
    pred_team_2      VARCHAR(250)   NOT NULL,
    pred_team_2_seed TINYINT        DEFAULT NULL,

    -- Model probabilities: P(pred_team_1 wins)
    prob_lgb         DECIMAL(6,4)   DEFAULT NULL,
    prob_xgb         DECIMAL(6,4)   DEFAULT NULL,
    prob_logistic    DECIMAL(6,4)   DEFAULT NULL,
    prob_ensemble    DECIMAL(6,4)   DEFAULT NULL,

    -- Your predicted winner
    predicted_winner      VARCHAR(250)  DEFAULT NULL,
    predicted_winner_seed TINYINT       DEFAULT NULL,

    -- Reality (filled in by update_bracket_results.py after games are played)
    game_id          VARCHAR(250)   DEFAULT NULL,   -- FK to games table
    actual_team_1    VARCHAR(250)   DEFAULT NULL,   -- who actually played
    actual_team_2    VARCHAR(250)   DEFAULT NULL,
    actual_winner    VARCHAR(250)   DEFAULT NULL,   -- actual game winner
    correct          TINYINT(1)     DEFAULT NULL,   -- NULL=pending, 1=correct, 0=wrong

    created_at       TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP
                                    ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE  KEY uk_slot        (bracket_year, bracket_slot),
    INDEX        idx_game_id   (game_id),
    INDEX        idx_round     (round),
    INDEX        idx_year      (bracket_year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def main():
    print("Creating bracket_predictions table...")
    conn = create_connection()
    if not conn:
        print("ERROR: could not connect to database")
        sys.exit(1)

    cursor = conn.cursor()
    try:
        cursor.execute(CREATE_TABLE_SQL)
        conn.commit()
        print("✅ Table 'bracket_predictions' created (or already exists)")
    except Exception as e:
        print(f"ERROR: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

    print("\nSlot naming convention:")
    print("  East_R64_G1   = East region, Round of 64, Game 1 (1-seed vs 16-seed)")
    print("  East_R32_G1   = East region, Round of 32, Game 1")
    print("  East_S16_G1   = East region, Sweet 16, Game 1")
    print("  East_E8_G1    = East region, Elite 8, Game 1")
    print("  FF_G1         = Final Four, Game 1")
    print("  Championship  = Championship game")
    print("\nNext: run marchmadness/predict_bracket.py to populate predictions")


if __name__ == "__main__":
    main()
