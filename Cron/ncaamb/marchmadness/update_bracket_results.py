#!/usr/bin/env python3
"""
Update bracket_predictions with actual tournament results.

Scans the games table for NCAA Tournament games in March/April 2026,
matches them to bracket_predictions rows by team names,
and fills in game_id, actual_team_1, actual_team_2, actual_winner, correct.

Run from ncaamb/ directory:
    python marchmadness/update_bracket_results.py
"""

import csv
import sys
from pathlib import Path
from datetime import date

ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

from scrapes.sqlconn import create_connection, fetch as sql_fetch

BRACKET_YEAR = 2026
MAPPINGS_CSV = Path(__file__).parent / "bracket_team_mappings.csv"


def load_team_mappings() -> dict:
    """Load bracket_name -> my_team_name mapping from CSV."""
    mapping = {}
    with open(MAPPINGS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            bracket_name = row["bracket_name"].strip()
            my_team_name = row["my_team_name"].strip()
            if bracket_name and my_team_name:
                mapping[bracket_name] = my_team_name
    return mapping


def fetch_tournament_games(conn, ml_team_names: set) -> list:
    """Fetch moneyline rows where either team is a tournament team."""
    placeholders = ", ".join(["%s"] * len(ml_team_names))
    names_list = list(ml_team_names)
    query = f"""
        SELECT game_id, game_date AS date,
               team_1, team_2,
               actual_score_team_1 AS team_1_score,
               actual_score_team_2 AS team_2_score,
               winning_team
        FROM moneyline
        WHERE season = %s
          AND game_date BETWEEN '2026-03-17' AND '2026-04-10'
          AND (team_1 IN ({placeholders}) OR team_2 IN ({placeholders}))
        ORDER BY game_date
    """
    return sql_fetch(conn, query, tuple([BRACKET_YEAR] + names_list + names_list))


def fetch_pending_predictions(conn) -> list:
    """Fetch all bracket slots not yet resolved (actual_winner is NULL)."""
    query = """
        SELECT id, bracket_slot, round, region,
               pred_team_1, pred_team_2, predicted_winner
        FROM bracket_predictions
        WHERE bracket_year = %s
          AND actual_winner IS NULL
        ORDER BY id
    """
    return sql_fetch(conn, query, (BRACKET_YEAR,))


def teams_match(pred_team: str, actual_team: str) -> bool:
    """Case-insensitive exact match."""
    return pred_team.strip().lower() == actual_team.strip().lower()


def find_matching_game(pred: dict, tournament_games: list, mapping: dict) -> dict | None:
    """
    Find the actual game that matches a prediction row.
    pred_team values are already my_team_names, same as moneyline team names.
    """
    t1 = pred["pred_team_1"]
    t2 = pred["pred_team_2"]
    for game in tournament_games:
        gt1 = game["team_1"]
        gt2 = game["team_2"]
        if (teams_match(t1, gt1) and teams_match(t2, gt2)) or \
           (teams_match(t1, gt2) and teams_match(t2, gt1)):
            return game
    return None


def update_result(conn, pred_id: int, game: dict, predicted_winner: str, mapping: dict):
    """Fill in actual result columns and set correct flag."""
    # Use winning_team if available, fall back to computing from scores
    if game.get("winning_team"):
        actual_winner = game["winning_team"]
    elif game["team_1_score"] is not None and game["team_2_score"] is not None:
        actual_winner = game["team_1"] if game["team_1_score"] > game["team_2_score"] else game["team_2"]
    else:
        return False  # no result yet

    # predicted_winner is already a my_team_name — compare directly
    correct = 1 if teams_match(predicted_winner, actual_winner) else 0

    sql = """
        UPDATE bracket_predictions
        SET game_id       = %s,
            actual_team_1 = %s,
            actual_team_2 = %s,
            actual_winner = %s,
            correct       = %s
        WHERE id = %s
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (
            game["game_id"],
            game["team_1"],
            game["team_2"],
            actual_winner,
            correct,
            pred_id,
        ))
        conn.commit()
    finally:
        cursor.close()
    return True


def main():
    print("\n" + "=" * 70)
    print("UPDATE BRACKET RESULTS — 2026")
    print("=" * 70 + "\n")

    conn = create_connection()
    if not conn:
        print("ERROR: could not connect to database")
        sys.exit(1)

    mapping = load_team_mappings()
    ml_team_names = set(mapping.values())
    print(f"Loaded {len(mapping)} team mappings from CSV\n")

    print("Fetching tournament games from moneyline table...")
    tournament_games = fetch_tournament_games(conn, ml_team_names)
    print(f"  Found {len(tournament_games)} tournament games\n")

    print("Fetching pending predictions...")
    pending = fetch_pending_predictions(conn)
    print(f"  Found {len(pending)} slots still pending\n")

    if not tournament_games:
        print("No tournament games in DB yet — nothing to match.")
        conn.close()
        return

    matched   = 0
    updated   = 0
    no_result = 0
    unmatched = []

    for pred in pending:
        game = find_matching_game(pred, tournament_games, mapping)
        if game is None:
            unmatched.append(pred["bracket_slot"])
            continue

        matched += 1
        result_saved = update_result(conn, pred["id"], game, pred["predicted_winner"], mapping)
        if result_saved:
            updated += 1
            actual_winner = (game.get("winning_team") or
                             (game["team_1"] if (game["team_1_score"] or 0) > (game["team_2_score"] or 0)
                              else game["team_2"]))
            correct_str = "✅" if teams_match(pred["predicted_winner"], actual_winner) else "❌"
            print(f"  {pred['bracket_slot']:25s}  "
                  f"{game['team_1']} vs {game['team_2']}  →  "
                  f"actual winner: {actual_winner}  "
                  f"{correct_str}")
        else:
            no_result += 1
            print(f"  {pred['bracket_slot']:25s}  matched game found but no scores yet")

    conn.close()

    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Pending slots:       {len(pending)}")
    print(f"  Matched to game:     {matched}")
    print(f"  Updated with result: {updated}")
    print(f"  Matched, no score:   {no_result}")
    if unmatched:
        print(f"\n  ⚠  No game found for {len(unmatched)} slot(s):")
        for s in unmatched:
            print(f"       {s}")
    print()

    # Print current standings
    print_standings(create_connection())


def print_standings(conn):
    """Print current bracket prediction accuracy."""
    if not conn:
        return
    query = """
        SELECT
            round,
            COUNT(*) AS total,
            SUM(CASE WHEN correct IS NOT NULL THEN 1 ELSE 0 END) AS played,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_picks
        FROM bracket_predictions
        WHERE bracket_year = %s
        GROUP BY round
        ORDER BY MIN(id)
    """
    rows = sql_fetch(conn, query, (BRACKET_YEAR,))
    conn.close()

    print("── Bracket standings ─────────────────────────────────")
    total_correct = 0
    total_played  = 0
    for r in rows:
        correct = int(r["correct_picks"] or 0)
        played  = int(r["played"] or 0)
        pct = (correct / played * 100) if played else 0
        print(f"  {str(r['round']):15s}  {correct:2d}/{played:2d} played  ({pct:.0f}%)")
        total_correct += correct
        total_played  += played

    overall = (total_correct / total_played * 100) if total_played else 0
    print(f"  {'OVERALL':15s}  {total_correct:2d}/{total_played:2d} played  ({overall:.0f}%)")


if __name__ == "__main__":
    main()
