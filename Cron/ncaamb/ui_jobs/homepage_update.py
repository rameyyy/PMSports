"""
Homepage Stats Update Script

Runs daily to pre-compute homepage statistics and insert into homepage_stats table.
This avoids complex queries on every page load.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scrapes.sqlconn import create_connection, fetch, execute_query
from datetime import datetime, timedelta
import pytz


def get_cst_dates():
    """Get today and yesterday dates in CST timezone"""
    cst = pytz.timezone('America/Chicago')
    now = datetime.now(cst)
    today = now.strftime('%Y-%m-%d')
    yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
    return today, yesterday


def get_todays_games_count(conn, date):
    """Get count of games for a specific date"""
    query = """
        SELECT COUNT(*) as count
        FROM ncaamb.games
        WHERE date = %s
    """
    result = fetch(conn, query, (date,))
    return result[0]['count'] if result else 0


def get_model_accuracy(conn, before_date):
    """Get my model accuracy stats (based on gbm probabilities) for games before given date"""
    query = """
        SELECT
            COUNT(*) as total_predictions,
            SUM(CASE
                WHEN (gbm_prob_team_1 > gbm_prob_team_2 AND winning_team = team_1)
                  OR (gbm_prob_team_1 <= gbm_prob_team_2 AND winning_team = team_2)
                THEN 1
                ELSE 0
            END) as correct_predictions
        FROM ncaamb.moneyline
        WHERE season = 2026
          AND winning_team IS NOT NULL
          AND gbm_prob_team_1 IS NOT NULL
          AND gbm_prob_team_2 IS NOT NULL
          AND best_book_odds_team_1 IS NOT NULL
          AND best_book_odds_team_2 IS NOT NULL
          AND game_date < %s
    """
    result = fetch(conn, query, (before_date,))

    if result and result[0]['total_predictions'] > 0:
        total = int(result[0]['total_predictions'])
        correct = int(result[0]['correct_predictions'])
        accuracy = round(correct / total * 100, 2)
        return accuracy, correct, total
    return 0.0, 0, 0


def get_vegas_accuracy(conn, before_date):
    """Get Vegas accuracy stats (based on moneyline favorite) for games before given date"""
    query = """
        SELECT
            COUNT(*) as total_predictions,
            SUM(CASE
                WHEN (best_book_odds_team_1 < best_book_odds_team_2 AND winning_team = team_1)
                  OR (best_book_odds_team_1 > best_book_odds_team_2 AND winning_team = team_2)
                THEN 1
                ELSE 0
            END) as correct_predictions
        FROM ncaamb.moneyline
        WHERE season = 2026
          AND winning_team IS NOT NULL
          AND best_book_odds_team_1 IS NOT NULL
          AND best_book_odds_team_2 IS NOT NULL
          AND game_date < %s
    """
    result = fetch(conn, query, (before_date,))

    if result and result[0]['total_predictions'] > 0:
        total = int(result[0]['total_predictions'])
        correct = int(result[0]['correct_predictions'])
        accuracy = round(correct / total * 100, 2)
        return accuracy, correct, total
    return 0.0, 0, 0


def american_to_decimal(odds):
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def get_pick_of_day_record(conn, before_date):
    """Get pick of day accuracy, ROI, and avg odds for PODs before given date"""
    record_query = """
        SELECT
            COUNT(*) as total_picks,
            SUM(CASE
                WHEN (m.gbm_prob_team_1 > m.gbm_prob_team_2 AND m.winning_team = m.team_1)
                  OR (m.gbm_prob_team_1 <= m.gbm_prob_team_2 AND m.winning_team = m.team_2)
                THEN 1
                ELSE 0
            END) as correct_picks
        FROM ncaamb.moneyline_picks mp
        JOIN ncaamb.moneyline m ON mp.game_id = m.game_id
        WHERE mp.pick_of_day = 1
          AND m.season = 2026
          AND m.winning_team IS NOT NULL
          AND mp.date < %s
    """
    record_result = fetch(conn, record_query, (before_date,))

    if record_result and record_result[0]['total_picks']:
        total_picks = int(record_result[0]['total_picks'])
        correct_picks = int(record_result[0]['correct_picks']) if record_result[0]['correct_picks'] else 0
        accuracy = round(correct_picks / total_picks * 100, 2) if total_picks > 0 else 0.0
    else:
        total_picks = 0
        correct_picks = 0
        accuracy = 0.0

    roi_query = """
        SELECT
            m.gbm_prob_team_1,
            m.gbm_prob_team_2,
            m.best_book_odds_team_1,
            m.best_book_odds_team_2,
            m.team_1,
            m.team_2,
            m.winning_team
        FROM ncaamb.moneyline_picks mp
        JOIN ncaamb.moneyline m ON mp.game_id = m.game_id
        WHERE mp.pick_of_day = 1
          AND m.season = 2026
          AND m.winning_team IS NOT NULL
          AND m.best_book_odds_team_1 IS NOT NULL
          AND m.best_book_odds_team_2 IS NOT NULL
          AND mp.date < %s
    """
    roi_data = fetch(conn, roi_query, (before_date,))

    total_wagered = 0
    total_profit = 0
    decimal_odds_list = []

    if roi_data:
        for row in roi_data:
            picked_team = row['team_1'] if row['gbm_prob_team_1'] > row['gbm_prob_team_2'] else row['team_2']
            picked_odds = int(row['best_book_odds_team_1']) if row['gbm_prob_team_1'] > row['gbm_prob_team_2'] else int(row['best_book_odds_team_2'])

            decimal_odds_list.append(american_to_decimal(picked_odds))

            stake = 100
            total_wagered += stake

            if picked_team == row['winning_team']:
                decimal_odds = american_to_decimal(picked_odds)
                profit = stake * (decimal_odds - 1)
                total_profit += profit
            else:
                total_profit -= stake

    roi = round(total_profit / total_wagered * 100, 2) if total_wagered > 0 else 0.0
    avg_decimal_odds = sum(decimal_odds_list) / len(decimal_odds_list) if decimal_odds_list else 1.0
    avg_american_odds = decimal_to_american(avg_decimal_odds) if decimal_odds_list else 0

    return accuracy, correct_picks, total_picks, avg_american_odds, roi


def get_pick_details(conn, date):
    """Get pick of day details for a specific date"""
    query = """
        SELECT
            mp.game_id,
            mp.betting_rule,
            m.team_1,
            m.team_2,
            m.gbm_prob_team_1,
            m.gbm_prob_team_2,
            m.best_book_odds_team_1,
            m.best_book_odds_team_2,
            m.winning_team
        FROM ncaamb.moneyline_picks mp
        JOIN ncaamb.moneyline m ON mp.game_id = m.game_id
        WHERE mp.pick_of_day = 1
          AND mp.date = %s
    """
    result = fetch(conn, query, (date,))

    if not result:
        return None, None, None, None

    pick = result[0]
    matchup = f"{pick['team_1']} vs {pick['team_2']}"

    if pick['gbm_prob_team_1'] > pick['gbm_prob_team_2']:
        picked_team = pick['team_1']
        picked_odds = int(pick['best_book_odds_team_1']) if pick['best_book_odds_team_1'] else None
    else:
        picked_team = pick['team_2']
        picked_odds = int(pick['best_book_odds_team_2']) if pick['best_book_odds_team_2'] else None

    outcome = None
    if pick['winning_team']:
        if pick['gbm_prob_team_1'] > pick['gbm_prob_team_2']:
            won = pick['winning_team'] == pick['team_1']
        else:
            won = pick['winning_team'] == pick['team_2']
        outcome = 'W' if won else 'L'

    return matchup, picked_team, picked_odds, outcome


def insert_homepage_stats(conn, stats):
    """Insert or update homepage stats"""
    query = """
        INSERT INTO ncaamb.homepage_stats (
            date, todays_games_count, my_accuracy, vegas_accuracy,
            my_total_correct, vegas_total_correct, total_complete_matches,
            pick_of_day_acc, pick_of_day_correct, pick_of_day_total,
            pod_avg_odds, pod_roi,
            pod_td_matchup, pod_td_pick, pod_td_odds,
            pod_yd_matchup, pod_yd_pick, pod_yd_odds, pod_yd_outcome
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            todays_games_count = VALUES(todays_games_count),
            my_accuracy = VALUES(my_accuracy),
            vegas_accuracy = VALUES(vegas_accuracy),
            my_total_correct = VALUES(my_total_correct),
            vegas_total_correct = VALUES(vegas_total_correct),
            total_complete_matches = VALUES(total_complete_matches),
            pick_of_day_acc = VALUES(pick_of_day_acc),
            pick_of_day_correct = VALUES(pick_of_day_correct),
            pick_of_day_total = VALUES(pick_of_day_total),
            pod_avg_odds = VALUES(pod_avg_odds),
            pod_roi = VALUES(pod_roi),
            pod_td_matchup = VALUES(pod_td_matchup),
            pod_td_pick = VALUES(pod_td_pick),
            pod_td_odds = VALUES(pod_td_odds),
            pod_yd_matchup = VALUES(pod_yd_matchup),
            pod_yd_pick = VALUES(pod_yd_pick),
            pod_yd_odds = VALUES(pod_yd_odds),
            pod_yd_outcome = VALUES(pod_yd_outcome)
    """

    params = (
        stats['date'],
        stats['todays_games_count'],
        stats['my_accuracy'],
        stats['vegas_accuracy'],
        stats['my_total_correct'],
        stats['vegas_total_correct'],
        stats['total_complete_matches'],
        stats['pick_of_day_acc'],
        stats['pick_of_day_correct'],
        stats['pick_of_day_total'],
        stats['pod_avg_odds'],
        stats['pod_roi'],
        stats['pod_td_matchup'],
        stats['pod_td_pick'],
        stats['pod_td_odds'],
        stats['pod_yd_matchup'],
        stats['pod_yd_pick'],
        stats['pod_yd_odds'],
        stats['pod_yd_outcome']
    )

    return execute_query(conn, query, params)


def main():
    print("=" * 80)
    print("HOMEPAGE STATS UPDATE")
    print("=" * 80)
    print()

    today, yesterday = get_cst_dates()
    print(f"Date: {today}")
    print(f"Yesterday: {yesterday}")
    print()

    conn = create_connection()
    if not conn:
        print("ERROR: Failed to connect to database")
        return

    try:
        print("Fetching stats...")
        print("-" * 40)

        todays_games = get_todays_games_count(conn, today)
        my_acc, my_correct, my_total = get_model_accuracy(conn, today)
        vegas_acc, vegas_correct, vegas_total = get_vegas_accuracy(conn, today)
        pod_acc, pod_correct, pod_total, pod_avg_odds, pod_roi = get_pick_of_day_record(conn, today)

        td_matchup, td_pick, td_odds, _ = get_pick_details(conn, today)
        yd_matchup, yd_pick, yd_odds, yd_outcome = get_pick_details(conn, yesterday)

        stats = {
            'date': today,
            'todays_games_count': todays_games,
            'my_accuracy': my_acc,
            'vegas_accuracy': vegas_acc,
            'my_total_correct': my_correct,
            'vegas_total_correct': vegas_correct,
            'total_complete_matches': my_total,
            'pick_of_day_acc': pod_acc,
            'pick_of_day_correct': pod_correct,
            'pick_of_day_total': pod_total,
            'pod_avg_odds': pod_avg_odds,
            'pod_roi': pod_roi,
            'pod_td_matchup': td_matchup,
            'pod_td_pick': td_pick,
            'pod_td_odds': td_odds,
            'pod_yd_matchup': yd_matchup,
            'pod_yd_pick': yd_pick,
            'pod_yd_odds': yd_odds,
            'pod_yd_outcome': yd_outcome
        }

        print(f"Today's games: {todays_games}")
        print(f"My accuracy: {my_acc}% ({my_correct}/{my_total})")
        print(f"Vegas accuracy: {vegas_acc}% ({vegas_correct}/{vegas_total})")
        print(f"POD record: {pod_acc}% ({pod_correct}/{pod_total})")
        print(f"POD avg odds: {pod_avg_odds}")
        print(f"POD ROI: {pod_roi}%")
        print(f"Today's POD: {td_pick} ({td_matchup}) @ {td_odds}")
        print(f"Yesterday's POD: {yd_pick} ({yd_matchup}) @ {yd_odds} -> {yd_outcome}")

        print()
        success = insert_homepage_stats(conn, stats)

        if success:
            print("SUCCESS: Homepage stats updated")
        else:
            print("ERROR: Failed to update homepage stats")

        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
