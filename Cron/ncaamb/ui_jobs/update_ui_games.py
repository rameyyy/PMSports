"""
Daily update script for ui_games table.
- Inserts new games for today
- Updates games missing actual results (actual_total, actual_winner)

Run as daily cron job.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapes.sqlconn import create_connection, fetch, execute_query
from datetime import datetime
import pytz


def get_today_cst():
    """Get today's date in CST timezone"""
    cst = pytz.timezone('America/Chicago')
    return datetime.now(cst).strftime('%Y-%m-%d')


def get_hna(location, team_1, team_2):
    """Determine Home/Neutral/Away status for team_1"""
    if location == 'N':
        return 'N'
    elif location == team_1:
        return 'H'
    elif location == team_2:
        return 'A'
    else:
        return 'N'


def upsert_game(conn, row):
    """Insert or update a single game"""
    hna = get_hna(row['location'], row['team_1'], row['team_2'])

    # Convert probabilities to percentages
    team_1_prob_algo = round(float(row['gbm_prob_team_1']) * 100, 1) if row['gbm_prob_team_1'] else None
    team_2_prob_algo = round(float(row['gbm_prob_team_2']) * 100, 1) if row['gbm_prob_team_2'] else None
    team_1_prob_vegas = round(float(row['implied_prob_team_1_devigged']) * 100, 1) if row['implied_prob_team_1_devigged'] else None
    team_2_prob_vegas = round(float(row['implied_prob_team_2_devigged']) * 100, 1) if row['implied_prob_team_2_devigged'] else None

    # Round total prediction
    total_pred = round(float(row['lgb_pred'])) if row['lgb_pred'] else None

    upsert_query = """
        INSERT INTO ui_games (
            game_id, date, team_1, team_1_hna, team_2,
            team_1_rank, team_2_rank,
            team_1_conference, team_2_conference,
            team_1_prob_algopicks, team_2_prob_algopicks,
            team_1_prob_vegas, team_2_prob_vegas,
            team_1_ml, team_2_ml,
            total_pred, vegas_ou_line,
            actual_total, actual_winner
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s
        )
        ON DUPLICATE KEY UPDATE
            date = VALUES(date),
            team_1 = VALUES(team_1),
            team_1_hna = VALUES(team_1_hna),
            team_2 = VALUES(team_2),
            team_1_rank = VALUES(team_1_rank),
            team_2_rank = VALUES(team_2_rank),
            team_1_conference = VALUES(team_1_conference),
            team_2_conference = VALUES(team_2_conference),
            team_1_prob_algopicks = VALUES(team_1_prob_algopicks),
            team_2_prob_algopicks = VALUES(team_2_prob_algopicks),
            team_1_prob_vegas = VALUES(team_1_prob_vegas),
            team_2_prob_vegas = VALUES(team_2_prob_vegas),
            team_1_ml = VALUES(team_1_ml),
            team_2_ml = VALUES(team_2_ml),
            total_pred = VALUES(total_pred),
            vegas_ou_line = VALUES(vegas_ou_line),
            actual_total = VALUES(actual_total),
            actual_winner = VALUES(actual_winner)
    """

    params = (
        row['game_id'],
        row['game_date'],
        row['team_1'],
        hna,
        row['team_2'],
        row['team_1_rank'],
        row['team_2_rank'],
        row['team_1_conference'],
        row['team_2_conference'],
        team_1_prob_algo,
        team_2_prob_algo,
        team_1_prob_vegas,
        team_2_prob_vegas,
        row['best_book_odds_team_1'],
        row['best_book_odds_team_2'],
        total_pred,
        float(row['over_point']) if row['over_point'] else None,
        row['actual_total'],
        row['winning_team']
    )

    cursor = conn.cursor()
    cursor.execute(upsert_query, params)
    affected = cursor.rowcount
    cursor.close()
    return affected


def get_games_query(where_clause):
    """Build the query to fetch game data"""
    return f"""
        SELECT
            m.game_id,
            m.game_date,
            m.team_1,
            m.team_2,
            g.location,
            l1.`rank` as team_1_rank,
            l2.`rank` as team_2_rank,
            COALESCE(t1_current.conference, t1_fallback.conference) as team_1_conference,
            COALESCE(t2_current.conference, t2_fallback.conference) as team_2_conference,
            m.gbm_prob_team_1,
            m.gbm_prob_team_2,
            m.implied_prob_team_1_devigged,
            m.implied_prob_team_2_devigged,
            m.best_book_odds_team_1,
            m.best_book_odds_team_2,
            o.lgb_pred,
            o.over_point,
            m.actual_total,
            m.winning_team,
            m.season
        FROM moneyline m
        LEFT JOIN overunder o ON m.game_id = o.game_id
        LEFT JOIN games g ON m.game_id = g.game_id
        LEFT JOIN leaderboard l1 ON l1.date = m.game_date AND l1.team = m.team_1
        LEFT JOIN leaderboard l2 ON l2.date = m.game_date AND l2.team = m.team_2
        LEFT JOIN teams t1_current ON t1_current.team_name = m.team_1 AND t1_current.season = m.season
        LEFT JOIN teams t1_fallback ON t1_fallback.team_name = m.team_1 AND t1_fallback.season = 2025
        LEFT JOIN teams t2_current ON t2_current.team_name = m.team_2 AND t2_current.season = m.season
        LEFT JOIN teams t2_fallback ON t2_fallback.team_name = m.team_2 AND t2_fallback.season = 2025
        {where_clause}
    """


def update_ui_games():
    """Main function to update ui_games table"""
    conn = create_connection()
    if not conn:
        print("Failed to connect to database")
        return False

    today = get_today_cst()
    print(f"Running ui_games update for {today}")

    insert_count = 0
    update_count = 0
    error_count = 0

    # 1. Get and insert/update today's games
    print("Fetching today's games...")
    today_query = get_games_query(f"WHERE m.game_date = '{today}'")
    today_games = fetch(conn, today_query)
    print(f"Found {len(today_games)} games for today")

    for row in today_games:
        try:
            affected = upsert_game(conn, row)
            if affected == 1:
                insert_count += 1
            elif affected == 2:
                update_count += 1
        except Exception as e:
            print(f"Error processing {row['game_id']}: {e}")
            error_count += 1

    # 2. Update games missing actual results
    print("Fetching games missing actual results...")
    missing_query = get_games_query("""
        WHERE m.game_id IN (
            SELECT game_id FROM ui_games
            WHERE actual_total IS NULL OR actual_winner IS NULL
        )
        AND m.actual_total IS NOT NULL
    """)
    missing_games = fetch(conn, missing_query)
    print(f"Found {len(missing_games)} games to update with results")

    for row in missing_games:
        try:
            affected = upsert_game(conn, row)
            if affected == 2:
                update_count += 1
        except Exception as e:
            print(f"Error updating {row['game_id']}: {e}")
            error_count += 1

    conn.commit()
    conn.close()

    print(f"Completed: {insert_count} inserted, {update_count} updated, {error_count} errors")
    return True


if __name__ == "__main__":
    update_ui_games()
