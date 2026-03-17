#!/usr/bin/env python3
"""
March Madness 2026 — Full Bracket Predictions

Runs all 67 games (First Four → Championship), cascades winners round by round,
and saves predictions to the bracket_predictions DB table.

Run from ncaamb/ directory:
    python marchmadness/predict_bracket.py
"""

import sys
import pickle
from datetime import date as dt_date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb

marchmadness = Path(__file__).parent
ncaamb_dir   = marchmadness.parent
sys.path.insert(0, str(ncaamb_dir))
sys.path.insert(0, str(marchmadness))

from models.build_flat_df import build_flat_row_for_game
from models.utils import fetch_games, fetch_leaderboard, fetch_player_stats
from models.overunder.build_ou_features import build_ou_features
from scrapes.sqlconn import create_connection, fetch as sql_fetch
from data_utils import ODDS_COLS, METADATA_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────
LGB_DIR = marchmadness / "models" / "lightgbm"
XGB_DIR = marchmadness / "models" / "xgboost"
LOG_DIR = marchmadness / "models" / "ridge"

# ── Ensemble weights (AUC-proportional) ───────────────────────────────────────
LGB_AUC = 0.796
XGB_AUC = 0.797
LOG_AUC = 0.787
_total  = LGB_AUC + XGB_AUC + LOG_AUC
W_LGB   = LGB_AUC / _total
W_XGB   = XGB_AUC / _total
W_LOG   = LOG_AUC / _total

BRACKET_YEAR = 2026
SEASON       = 2026

# ── Full bracket: (slot, round, region, team1, seed1, team2, seed2) ───────────
# team can be a name or "winner_of:SLOT" for later rounds
BRACKET = [
    # ── First Four ────────────────────────────────────────────────────────────
    ("East_FF_G1",    "First Four", "East",    "UMBC",             16, "Howard",             16),
    ("South_FF_G1",   "First Four", "South",   "Prairie View A&M", 16, "Lehigh",             16),
    ("West_FF_G1",    "First Four", "West",    "Texas",            11, "NC State",           11),
    ("Midwest_FF_G1", "First Four", "Midwest", "Miami (OH)",       11, "SMU",                11),

    # ── East — Round of 64 ────────────────────────────────────────────────────
    ("East_R64_G1", "First Round", "East", "Duke",           1, "Siena",              16),
    ("East_R64_G2", "First Round", "East", "Ohio State",     8, "TCU",                 9),
    ("East_R64_G3", "First Round", "East", "St. John's",     5, "Northern Iowa",      12),
    ("East_R64_G4", "First Round", "East", "Kansas",         4, "Cal Baptist",        13),
    ("East_R64_G5", "First Round", "East", "Louisville",     6, "South Florida",      11),
    ("East_R64_G6", "First Round", "East", "Michigan State", 3, "North Dakota State", 14),
    ("East_R64_G7", "First Round", "East", "UCLA",           7, "UCF",                10),
    ("East_R64_G8", "First Round", "East", "UConn",          2, "Furman",             15),

    # ── South — Round of 64 ───────────────────────────────────────────────────
    ("South_R64_G1", "First Round", "South", "Florida",        1, "winner_of:South_FF_G1", 16),
    ("South_R64_G2", "First Round", "South", "Clemson",        8, "Iowa",                   9),
    ("South_R64_G3", "First Round", "South", "Vanderbilt",     5, "McNeese",               12),
    ("South_R64_G4", "First Round", "South", "Nebraska",       4, "Troy",                  13),
    ("South_R64_G5", "First Round", "South", "North Carolina", 6, "VCU",                   11),
    ("South_R64_G6", "First Round", "South", "Illinois",       3, "Penn",                  14),
    ("South_R64_G7", "First Round", "South", "Saint Mary's",   7, "Texas A&M",             10),
    ("South_R64_G8", "First Round", "South", "Houston",        2, "Idaho",                 15),

    # ── West — Round of 64 ────────────────────────────────────────────────────
    ("West_R64_G1", "First Round", "West", "Arizona",     1, "Long Island",           16),
    ("West_R64_G2", "First Round", "West", "Villanova",   8, "Utah State",             9),
    ("West_R64_G3", "First Round", "West", "Wisconsin",   5, "High Point",            12),
    ("West_R64_G4", "First Round", "West", "Arkansas",    4, "Hawaii",                13),
    ("West_R64_G5", "First Round", "West", "BYU",         6, "winner_of:West_FF_G1",  11),
    ("West_R64_G6", "First Round", "West", "Gonzaga",     3, "Kennesaw State",        14),
    ("West_R64_G7", "First Round", "West", "Miami (FL)",  7, "Missouri",              10),
    ("West_R64_G8", "First Round", "West", "Purdue",      2, "Queens (NC)",           15),

    # ── Midwest — Round of 64 ─────────────────────────────────────────────────
    ("Midwest_R64_G1", "First Round", "Midwest", "Michigan",  1, "winner_of:East_FF_G1",    16),
    ("Midwest_R64_G2", "First Round", "Midwest", "Georgia",   8, "Saint Louis",              9),
    ("Midwest_R64_G3", "First Round", "Midwest", "Texas Tech",5, "Akron",                   12),
    ("Midwest_R64_G4", "First Round", "Midwest", "Alabama",   4, "Hofstra",                 13),
    ("Midwest_R64_G5", "First Round", "Midwest", "Tennessee", 6, "winner_of:Midwest_FF_G1", 11),
    ("Midwest_R64_G6", "First Round", "Midwest", "Virginia",  3, "Wright State",            14),
    ("Midwest_R64_G7", "First Round", "Midwest", "Kentucky",  7, "Santa Clara",             10),
    ("Midwest_R64_G8", "First Round", "Midwest", "Iowa State",2, "Tennessee State",         15),

    # ── Round of 32 ───────────────────────────────────────────────────────────
    ("East_R32_G1",    "Second Round", "East",    "winner_of:East_R64_G1",    None, "winner_of:East_R64_G2",    None),
    ("East_R32_G2",    "Second Round", "East",    "winner_of:East_R64_G3",    None, "winner_of:East_R64_G4",    None),
    ("East_R32_G3",    "Second Round", "East",    "winner_of:East_R64_G5",    None, "winner_of:East_R64_G6",    None),
    ("East_R32_G4",    "Second Round", "East",    "winner_of:East_R64_G7",    None, "winner_of:East_R64_G8",    None),
    ("South_R32_G1",   "Second Round", "South",   "winner_of:South_R64_G1",   None, "winner_of:South_R64_G2",   None),
    ("South_R32_G2",   "Second Round", "South",   "winner_of:South_R64_G3",   None, "winner_of:South_R64_G4",   None),
    ("South_R32_G3",   "Second Round", "South",   "winner_of:South_R64_G5",   None, "winner_of:South_R64_G6",   None),
    ("South_R32_G4",   "Second Round", "South",   "winner_of:South_R64_G7",   None, "winner_of:South_R64_G8",   None),
    ("West_R32_G1",    "Second Round", "West",    "winner_of:West_R64_G1",    None, "winner_of:West_R64_G2",    None),
    ("West_R32_G2",    "Second Round", "West",    "winner_of:West_R64_G3",    None, "winner_of:West_R64_G4",    None),
    ("West_R32_G3",    "Second Round", "West",    "winner_of:West_R64_G5",    None, "winner_of:West_R64_G6",    None),
    ("West_R32_G4",    "Second Round", "West",    "winner_of:West_R64_G7",    None, "winner_of:West_R64_G8",    None),
    ("Midwest_R32_G1", "Second Round", "Midwest", "winner_of:Midwest_R64_G1", None, "winner_of:Midwest_R64_G2", None),
    ("Midwest_R32_G2", "Second Round", "Midwest", "winner_of:Midwest_R64_G3", None, "winner_of:Midwest_R64_G4", None),
    ("Midwest_R32_G3", "Second Round", "Midwest", "winner_of:Midwest_R64_G5", None, "winner_of:Midwest_R64_G6", None),
    ("Midwest_R32_G4", "Second Round", "Midwest", "winner_of:Midwest_R64_G7", None, "winner_of:Midwest_R64_G8", None),

    # ── Sweet 16 ──────────────────────────────────────────────────────────────
    ("East_S16_G1",    "Sweet 16", "East",    "winner_of:East_R32_G1",    None, "winner_of:East_R32_G2",    None),
    ("East_S16_G2",    "Sweet 16", "East",    "winner_of:East_R32_G3",    None, "winner_of:East_R32_G4",    None),
    ("South_S16_G1",   "Sweet 16", "South",   "winner_of:South_R32_G1",   None, "winner_of:South_R32_G2",   None),
    ("South_S16_G2",   "Sweet 16", "South",   "winner_of:South_R32_G3",   None, "winner_of:South_R32_G4",   None),
    ("West_S16_G1",    "Sweet 16", "West",    "winner_of:West_R32_G1",    None, "winner_of:West_R32_G2",    None),
    ("West_S16_G2",    "Sweet 16", "West",    "winner_of:West_R32_G3",    None, "winner_of:West_R32_G4",    None),
    ("Midwest_S16_G1", "Sweet 16", "Midwest", "winner_of:Midwest_R32_G1", None, "winner_of:Midwest_R32_G2", None),
    ("Midwest_S16_G2", "Sweet 16", "Midwest", "winner_of:Midwest_R32_G3", None, "winner_of:Midwest_R32_G4", None),

    # ── Elite 8 ───────────────────────────────────────────────────────────────
    ("East_E8_G1",    "Elite 8", "East",    "winner_of:East_S16_G1",    None, "winner_of:East_S16_G2",    None),
    ("South_E8_G1",   "Elite 8", "South",   "winner_of:South_S16_G1",   None, "winner_of:South_S16_G2",   None),
    ("West_E8_G1",    "Elite 8", "West",    "winner_of:West_S16_G1",    None, "winner_of:West_S16_G2",    None),
    ("Midwest_E8_G1", "Elite 8", "Midwest", "winner_of:Midwest_S16_G1", None, "winner_of:Midwest_S16_G2", None),

    # ── Final Four ────────────────────────────────────────────────────────────
    ("FF_G1", "Final Four", None, "winner_of:East_E8_G1",    None, "winner_of:West_E8_G1",    None),
    ("FF_G2", "Final Four", None, "winner_of:South_E8_G1",   None, "winner_of:Midwest_E8_G1", None),

    # ── Championship ──────────────────────────────────────────────────────────
    ("Championship", "Championship", None, "winner_of:FF_G1", None, "winner_of:FF_G2", None),
]


def load_team_map() -> dict:
    """bracket_name → internal DB team name"""
    df = pl.read_csv(str(marchmadness / "bracket_team_mappings.csv"))
    return dict(zip(df["bracket_name"].to_list(), df["my_team_name"].to_list()))


def load_models(feature_cols: list):
    """Load all three trained models."""
    print("  Loading LightGBM...")
    lgb_model = lgb.Booster(model_file=str(LGB_DIR / "lightgbm_model.txt"))

    print("  Loading XGBoost...")
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    xgb_model.load_model(str(XGB_DIR / "xgboost_model.json"))

    print("  Loading Logistic Regression...")
    with open(LOG_DIR / "logistic_model.pkl", "rb") as f:
        log_bundle = pickle.load(f)
    log_model  = log_bundle["model"]
    log_scaler = log_bundle["scaler"]

    return lgb_model, xgb_model, log_model, log_scaler


def load_feature_cols() -> list:
    """All three models use identical feature columns — load from LGB dir."""
    return (LGB_DIR / "feature_columns.txt").read_text().strip().splitlines()


def get_pred_date(leaderboard_df: pl.DataFrame) -> dt_date:
    """Use max leaderboard date + 1 so the lookup always finds data."""
    max_lb_date = leaderboard_df["date"].max()
    return max_lb_date + timedelta(days=1)


def check_team_data(
    team_map: dict,
    games_df: pl.DataFrame,
    leaderboard_df: pl.DataFrame,
    pred_date: dt_date,
) -> set:
    """
    For every named team in the bracket, check:
      - how many 2026 games exist in games_df
      - whether leaderboard has an entry on pred_date - 1

    Prints a report and returns the set of internal team names that have
    NO game history (zero rows) — these will produce near-zero features.
    """
    lb_date = pred_date - timedelta(days=1)
    lb_teams = set(leaderboard_df.filter(pl.col("date") == lb_date)["team"].to_list())

    # Collect all named (non-winner_of) teams from bracket
    named_teams: set[str] = set()
    for (_, _, _, t1, _, t2, _) in BRACKET:
        for t in (t1, t2):
            if isinstance(t, str) and not t.startswith("winner_of:"):
                named_teams.add(t)

    print(f"\n{'Team':<30} {'2026 games':>10}  {'Leaderboard':>12}")
    print("-" * 58)

    no_games: set[str] = set()
    no_lb:    set[str] = set()

    for bracket_name in sorted(named_teams):
        internal = team_map.get(bracket_name, bracket_name)
        n_games = len(games_df.filter(
            (pl.col("team_1") == internal) | (pl.col("team_2") == internal)
        ))
        has_lb = internal in lb_teams
        lb_str = f"✅ {lb_date}" if has_lb else f"❌ none on {lb_date}"

        if n_games == 0:
            no_games.add(internal)
        if not has_lb:
            no_lb.add(internal)

        flag = " ⚠" if (n_games == 0 or not has_lb) else ""
        print(f"  {bracket_name:<28} {n_games:>10}  {lb_str}{flag}")

    print()
    if no_games:
        print(f"  ⚠  {len(no_games)} team(s) have NO 2026 game history — features will be null/0:")
        for t in sorted(no_games):
            print(f"       {t}")
    if no_lb:
        print(f"  ⚠  {len(no_lb)} team(s) have NO leaderboard entry on {lb_date}:")
        for t in sorted(no_lb):
            print(f"       {t}")
    if not no_games and not no_lb:
        print("  ✅ All teams have game history and leaderboard data.")
    print()

    return no_games


def predict_game(
    team_1: str,
    team_2: str,
    pred_date: dt_date,
    games_df: pl.DataFrame,
    leaderboard_df: pl.DataFrame,
    player_stats_df: pl.DataFrame,
    feature_cols: list,
    lgb_model,
    xgb_model,
    log_model,
    log_scaler,
    slot: str,
) -> dict:
    """
    Build features for one game and return probability dict.
    Returns {"lgb": float, "xgb": float, "log": float, "ensemble": float}
    """
    game_row = {
        "game_id":            f"mm{BRACKET_YEAR}_{slot}",
        "season":             SEASON,
        "date":               pred_date,
        "team_1":             team_1,
        "team_2":             team_2,
        "team_1_score":       None,
        "team_2_score":       None,
        "location":           "N",      # tournament = neutral site
        "game_type":          "NCAA Tournament",
        "team_1_conference":  None,
        "team_2_conference":  None,
    }

    try:
        flat_row = build_flat_row_for_game(
            game_row, games_df, leaderboard_df, player_stats_df, odds_dict={}
        )
    except Exception as e:
        print(f"    ⚠  build_flat_row_for_game failed for {team_1} vs {team_2}: {e}")
        return {"lgb": 0.5, "xgb": 0.5, "log": 0.5, "ensemble": 0.5}

    flat_row["game_odds"] = []

    try:
        raw_df      = pl.DataFrame([flat_row], infer_schema_length=None)
        features_df = build_ou_features(raw_df)
    except Exception as e:
        print(f"    ⚠  build_ou_features failed for {team_1} vs {team_2}: {e}")
        return {"lgb": 0.5, "xgb": 0.5, "log": 0.5, "ensemble": 0.5}

    # Align to training feature columns (fill missing as null → 0)
    for col in feature_cols:
        if col not in features_df.columns:
            features_df = features_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    X = features_df.select(feature_cols).fill_null(0).to_numpy()

    p_lgb = float(lgb_model.predict(X)[0])
    p_xgb = float(xgb_model.predict_proba(X)[0, 1])
    X_sc  = log_scaler.transform(X)
    p_log = float(log_model.predict_proba(X_sc)[0, 1])
    p_ens = W_LGB * p_lgb + W_XGB * p_xgb + W_LOG * p_log

    return {"lgb": p_lgb, "xgb": p_xgb, "log": p_log, "ensemble": p_ens}


def insert_prediction(conn, row: dict):
    """Upsert one prediction row into bracket_predictions."""
    sql = """
        INSERT INTO bracket_predictions
            (bracket_year, bracket_slot, round, region,
             pred_team_1, pred_team_1_seed, pred_team_2, pred_team_2_seed,
             prob_lgb, prob_xgb, prob_logistic, prob_ensemble,
             predicted_winner, predicted_winner_seed)
        VALUES
            (%(bracket_year)s, %(bracket_slot)s, %(round)s, %(region)s,
             %(pred_team_1)s, %(pred_team_1_seed)s, %(pred_team_2)s, %(pred_team_2_seed)s,
             %(prob_lgb)s, %(prob_xgb)s, %(prob_logistic)s, %(prob_ensemble)s,
             %(predicted_winner)s, %(predicted_winner_seed)s)
        ON DUPLICATE KEY UPDATE
            round                = VALUES(round),
            region               = VALUES(region),
            pred_team_1          = VALUES(pred_team_1),
            pred_team_1_seed     = VALUES(pred_team_1_seed),
            pred_team_2          = VALUES(pred_team_2),
            pred_team_2_seed     = VALUES(pred_team_2_seed),
            prob_lgb             = VALUES(prob_lgb),
            prob_xgb             = VALUES(prob_xgb),
            prob_logistic        = VALUES(prob_logistic),
            prob_ensemble        = VALUES(prob_ensemble),
            predicted_winner     = VALUES(predicted_winner),
            predicted_winner_seed = VALUES(predicted_winner_seed)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, row)
        conn.commit()
    finally:
        cursor.close()


def main():
    print("\n" + "=" * 70)
    print("MARCH MADNESS 2026 — BRACKET PREDICTIONS")
    print("=" * 70 + "\n")

    # ── Load team name mapping ────────────────────────────────────────────────
    team_map = load_team_map()

    # ── Load feature columns ──────────────────────────────────────────────────
    print("Loading feature columns...")
    feature_cols = load_feature_cols()
    print(f"  {len(feature_cols)} feature columns\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    lgb_model, xgb_model, log_model, log_scaler = load_models(feature_cols)
    print()

    # ── Load DB data ──────────────────────────────────────────────────────────
    print("Loading 2026 season data from DB...")
    games_df        = fetch_games(season=SEASON)
    leaderboard_df  = fetch_leaderboard()
    player_stats_df = fetch_player_stats(season=SEASON)

    # Only completed games for historical context
    games_df = games_df.filter(
        pl.col("team_1_score").is_not_null() &
        pl.col("team_2_score").is_not_null()
    )
    print(f"  Completed 2026 games (for context): {len(games_df)}")
    print(f"  Leaderboard rows: {len(leaderboard_df)}")

    pred_date = get_pred_date(leaderboard_df)
    print(f"  Prediction date (max lb + 1): {pred_date}\n")

    # ── Pre-flight: check every team has data ─────────────────────────────────
    print("=" * 70)
    print("PRE-FLIGHT DATA CHECK")
    print("=" * 70)
    teams_without_games = check_team_data(team_map, games_df, leaderboard_df, pred_date)

    if teams_without_games:
        print(f"  NOTE: {len(teams_without_games)} team(s) have no game history.")
        print("  Their predictions will use null features (filled with 0).")
        print("  Results will be less reliable for those matchups.\n")

    # ── DB connection ─────────────────────────────────────────────────────────
    conn = create_connection()
    if not conn:
        print("ERROR: could not connect to database")
        sys.exit(1)

    # ── Run bracket ───────────────────────────────────────────────────────────
    # slot_results: slot → (internal_name, seed)
    slot_results: dict[str, tuple] = {}
    total_games = len(BRACKET)
    print(f"Running predictions for {total_games} games...\n")

    for i, (slot, rnd, region, raw_t1, seed1, raw_t2, seed2) in enumerate(BRACKET, 1):
        # Resolve "winner_of:SLOT" references
        if isinstance(raw_t1, str) and raw_t1.startswith("winner_of:"):
            src_slot = raw_t1.split(":", 1)[1]
            if src_slot not in slot_results:
                print(f"  ⚠  {slot}: cannot resolve {raw_t1} — skipping")
                continue
            bracket_t1, seed1 = slot_results[src_slot]
        else:
            bracket_t1 = raw_t1

        if isinstance(raw_t2, str) and raw_t2.startswith("winner_of:"):
            src_slot = raw_t2.split(":", 1)[1]
            if src_slot not in slot_results:
                print(f"  ⚠  {slot}: cannot resolve {raw_t2} — skipping")
                continue
            bracket_t2, seed2 = slot_results[src_slot]
        else:
            bracket_t2 = raw_t2

        # Map bracket names → internal DB names
        int_t1 = team_map.get(bracket_t1, bracket_t1)
        int_t2 = team_map.get(bracket_t2, bracket_t2)

        warn = ""
        if int_t1 in teams_without_games or int_t2 in teams_without_games:
            warn = "  ⚠ sparse data"
        print(f"  [{i:02d}/{total_games}] {slot:25s}  {rnd:14s}  "
              f"{int_t1}({seed1}) vs {int_t2}({seed2}){warn}")

        probs = predict_game(
            int_t1, int_t2, pred_date,
            games_df, leaderboard_df, player_stats_df,
            feature_cols, lgb_model, xgb_model, log_model, log_scaler,
            slot,
        )

        p_ens = probs["ensemble"]
        lgb_pick1 = probs["lgb"] >= 0.5
        xgb_pick1 = probs["xgb"] >= 0.5
        log_pick1 = probs["log"] >= 0.5

        # LGB outvoted: XGB and LOG both disagree with LGB → go with majority
        if lgb_pick1 != xgb_pick1 and lgb_pick1 != log_pick1:
            pick_team1 = xgb_pick1  # xgb and log agree
            vote = "majority(xgb+log)"
        else:
            # LGB + at least one other agree → pick LGB
            # All agree → also LGB (ensemble prob used for display)
            pick_team1 = lgb_pick1
            vote = "lgb" if (xgb_pick1 == lgb_pick1 and log_pick1 == lgb_pick1) else "lgb+other"

        if pick_team1:
            winner, winner_seed = int_t1, seed1
        else:
            winner, winner_seed = int_t2, seed2

        print(f"           → {winner} wins  [{vote}]  "
              f"(lgb={probs['lgb']:.3f}  xgb={probs['xgb']:.3f}  "
              f"log={probs['log']:.3f}  ens={probs['ensemble']:.3f})")

        # Store for downstream rounds
        slot_results[slot] = (winner, winner_seed)

        # DB row — store internal team names and probabilities as P(team_1 wins)
        db_row = {
            "bracket_year":        BRACKET_YEAR,
            "bracket_slot":        slot,
            "round":               rnd,
            "region":              region,
            "pred_team_1":         int_t1,
            "pred_team_1_seed":    seed1,
            "pred_team_2":         int_t2,
            "pred_team_2_seed":    seed2,
            "prob_lgb":            round(probs["lgb"], 4),
            "prob_xgb":            round(probs["xgb"], 4),
            "prob_logistic":       round(probs["log"], 4),
            "prob_ensemble":       round(probs["ensemble"], 4),
            "predicted_winner":    winner,
            "predicted_winner_seed": winner_seed,
        }
        insert_prediction(conn, db_row)

    conn.close()

    print("\n" + "=" * 70)
    print("BRACKET COMPLETE")
    print("=" * 70)
    print(f"\nChampion prediction: {slot_results.get('Championship', ('?', '?'))[0]}")
    print(f"\nAll {len(slot_results)} games saved to bracket_predictions table.")
    print("\nNext: run update_bracket_results.py after games are played.")


if __name__ == "__main__":
    main()
