#!/usr/bin/env python3
"""
Predict FF_G1 (Connecticut vs Arizona) and Championship (winner vs Florida).
Reuses predict_bracket.py functions exactly.

Run from ncaamb/ directory:
    python marchmadness/predict_ff_champ.py
"""

import sys
from pathlib import Path
from datetime import timedelta
import polars as pl

marchmadness = Path(__file__).parent
ncaamb_dir   = marchmadness.parent
sys.path.insert(0, str(ncaamb_dir))
sys.path.insert(0, str(marchmadness))

# Import everything from predict_bracket — same models, same feature pipeline
from predict_bracket import (
    load_models, load_feature_cols, predict_game, insert_prediction,
    BRACKET_YEAR, SEASON,
)
from models.utils import fetch_games, fetch_leaderboard, fetch_player_stats
from scrapes.sqlconn import create_connection


FF_G1_TEAM1, FF_G1_SEED1 = "Connecticut", 2
FF_G1_TEAM2, FF_G1_SEED2 = "Arizona",     1
CHAMP_OPP,   CHAMP_OPP_S = "Florida",     1


def pick_winner(probs, t1, s1, t2, s2):
    lgb_p1 = probs["lgb"] >= 0.5
    xgb_p1 = probs["xgb"] >= 0.5
    log_p1 = probs["log"] >= 0.5
    if lgb_p1 != xgb_p1 and lgb_p1 != log_p1:
        pick1 = xgb_p1
    else:
        pick1 = lgb_p1
    return (t1, s1) if pick1 else (t2, s2)


def main():
    print("\n" + "=" * 60)
    print("FF_G1 + CHAMPIONSHIP PREDICTIONS")
    print("=" * 60 + "\n")

    feature_cols = load_feature_cols()
    print(f"Loading models... ({len(feature_cols)} features)")
    lgb_model, xgb_model, log_model, log_scaler = load_models(feature_cols)

    print("Loading DB data...")
    games_df        = fetch_games(season=SEASON)
    leaderboard_df  = fetch_leaderboard()
    player_stats_df = fetch_player_stats(season=SEASON)
    games_df        = games_df.filter(
        pl.col("team_1_score").is_not_null() & pl.col("team_2_score").is_not_null()
    )
    pred_date = leaderboard_df["date"].max() + timedelta(days=1)
    print(f"  Completed games: {len(games_df)},  pred_date: {pred_date}\n")

    conn = create_connection()
    if not conn:
        print("ERROR: could not connect to DB"); sys.exit(1)

    # ── FF_G1 ─────────────────────────────────────────────────────────────────
    print(f"[1/2] FF_G1: {FF_G1_TEAM1}({FF_G1_SEED1}) vs {FF_G1_TEAM2}({FF_G1_SEED2})")
    probs_ff = predict_game(
        FF_G1_TEAM1, FF_G1_TEAM2, pred_date,
        games_df, leaderboard_df, player_stats_df,
        feature_cols, lgb_model, xgb_model, log_model, log_scaler,
        "FF_G1",
    )
    ff_winner, ff_seed = pick_winner(probs_ff, FF_G1_TEAM1, FF_G1_SEED1, FF_G1_TEAM2, FF_G1_SEED2)
    print(f"  → {ff_winner}  lgb={probs_ff['lgb']:.3f} xgb={probs_ff['xgb']:.3f} "
          f"log={probs_ff['log']:.3f} ens={probs_ff['ensemble']:.3f}")

    insert_prediction(conn, {
        "bracket_year": BRACKET_YEAR, "bracket_slot": "FF_G1",
        "round": "Final Four", "region": None,
        "pred_team_1": FF_G1_TEAM1, "pred_team_1_seed": FF_G1_SEED1,
        "pred_team_2": FF_G1_TEAM2, "pred_team_2_seed": FF_G1_SEED2,
        "prob_lgb": round(probs_ff["lgb"], 4), "prob_xgb": round(probs_ff["xgb"], 4),
        "prob_logistic": round(probs_ff["log"], 4), "prob_ensemble": round(probs_ff["ensemble"], 4),
        "predicted_winner": ff_winner, "predicted_winner_seed": ff_seed,
    })
    print("  ✅ FF_G1 saved\n")

    # ── Championship ──────────────────────────────────────────────────────────
    print(f"[2/2] Championship: {ff_winner}({ff_seed}) vs {CHAMP_OPP}({CHAMP_OPP_S})")
    probs_ch = predict_game(
        ff_winner, CHAMP_OPP, pred_date,
        games_df, leaderboard_df, player_stats_df,
        feature_cols, lgb_model, xgb_model, log_model, log_scaler,
        "Championship",
    )
    ch_winner, ch_seed = pick_winner(probs_ch, ff_winner, ff_seed, CHAMP_OPP, CHAMP_OPP_S)
    print(f"  → {ch_winner}  lgb={probs_ch['lgb']:.3f} xgb={probs_ch['xgb']:.3f} "
          f"log={probs_ch['log']:.3f} ens={probs_ch['ensemble']:.3f}")

    insert_prediction(conn, {
        "bracket_year": BRACKET_YEAR, "bracket_slot": "Championship",
        "round": "Championship", "region": None,
        "pred_team_1": ff_winner, "pred_team_1_seed": ff_seed,
        "pred_team_2": CHAMP_OPP, "pred_team_2_seed": CHAMP_OPP_S,
        "prob_lgb": round(probs_ch["lgb"], 4), "prob_xgb": round(probs_ch["xgb"], 4),
        "prob_logistic": round(probs_ch["log"], 4), "prob_ensemble": round(probs_ch["ensemble"], 4),
        "predicted_winner": ch_winner, "predicted_winner_seed": ch_seed,
    })
    print("  ✅ Championship saved\n")

    conn.close()
    print("=" * 60)
    print(f"Champion prediction: {ch_winner}")
    print("=" * 60)


if __name__ == "__main__":
    main()
