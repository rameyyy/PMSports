"""
NCAA Men's Basketball (ncaamb) data models and database queries
"""
from utils.db import execute_query


def get_cst_date():
    from datetime import datetime, timedelta
    import pytz
    cst = pytz.timezone('America/Chicago')
    return datetime.now(cst)

def get_homepage_stats():
    from datetime import datetime
    import pytz

    cst = pytz.timezone('America/Chicago')
    current_time_cst = datetime.now(cst)
    hour = current_time_cst.hour
    minute = current_time_cst.minute

    is_blackout = (hour < 8) or (hour == 8 and minute < 15)

    if is_blackout:
        todays_games = -1
    else:
        today_cst = current_time_cst.strftime('%Y-%m-%d')
        games_query = """
            SELECT COUNT(*) as count
            FROM ncaamb.games
            WHERE date = %s
        """
        games_result = execute_query(games_query, (today_cst,), fetch_one=True, database='ncaamb')
        todays_games = games_result['count'] if games_result else 0

    # Get my model accuracy (based on gbm probabilities)
    model_accuracy_query = """
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
          AND best_book_odds_team_1 != best_book_odds_team_2
    """
    model_result = execute_query(model_accuracy_query, fetch_one=True, database='ncaamb')

    if model_result and model_result['total_predictions'] > 0:
        model_accuracy = float(model_result['correct_predictions']) / float(model_result['total_predictions']) * 100
        model_correct = int(model_result['correct_predictions'])
        model_total = int(model_result['total_predictions'])
    else:
        model_accuracy = 0.0
        model_correct = 0
        model_total = 0

    # Get Vegas accuracy (based on moneyline favorite)
    # In American odds: negative = favorite, positive = underdog
    vegas_accuracy_query = """
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
          AND best_book_odds_team_1 != best_book_odds_team_2
    """
    vegas_result = execute_query(vegas_accuracy_query, fetch_one=True, database='ncaamb')

    if vegas_result and vegas_result['total_predictions'] > 0:
        vegas_accuracy = float(vegas_result['correct_predictions']) / float(vegas_result['total_predictions']) * 100
        vegas_correct = int(vegas_result['correct_predictions'])
        vegas_total = int(vegas_result['total_predictions'])
    else:
        vegas_accuracy = 0.0
        vegas_correct = 0
        vegas_total = 0

    # Calculate edge
    edge = model_accuracy - vegas_accuracy

    return {
        'todays_games': todays_games,
        'model_accuracy': round(model_accuracy, 2),
        'model_correct': model_correct,
        'model_total': model_total,
        'vegas_accuracy': round(vegas_accuracy, 2),
        'vegas_correct': vegas_correct,
        'vegas_total': vegas_total,
        'edge': round(edge, 2)
    }


def get_pick_of_day_data():
    from datetime import datetime, timedelta
    import pytz

    cst = pytz.timezone('America/Chicago')
    current_time_cst = datetime.now(cst)
    today_cst = current_time_cst.strftime('%Y-%m-%d')
    yesterday_cst = (current_time_cst - timedelta(days=1)).strftime('%Y-%m-%d')

    today_pick_query = """
        SELECT
            mp.game_id,
            mp.betting_rule,
            m.team_1,
            m.team_2,
            m.gbm_prob_team_1,
            m.gbm_prob_team_2,
            m.best_book_odds_team_1,
            m.best_book_odds_team_2,
            m.winning_team,
            g.date,
            g.time
        FROM ncaamb.moneyline_picks mp
        JOIN ncaamb.moneyline m ON mp.game_id = m.game_id
        JOIN ncaamb.games g ON mp.game_id = g.game_id
        WHERE mp.pick_of_day = 1
          AND mp.date = %s
    """
    today_pick = execute_query(today_pick_query, (today_cst,), fetch_one=True, database='ncaamb')

    yesterday_pick_query = """
        SELECT
            mp.game_id,
            mp.betting_rule,
            m.team_1,
            m.team_2,
            m.gbm_prob_team_1,
            m.gbm_prob_team_2,
            m.best_book_odds_team_1,
            m.best_book_odds_team_2,
            m.winning_team,
            g.date,
            g.time
        FROM ncaamb.moneyline_picks mp
        JOIN ncaamb.moneyline m ON mp.game_id = m.game_id
        JOIN ncaamb.games g ON mp.game_id = g.game_id
        WHERE mp.pick_of_day = 1
          AND mp.date = %s
    """
    yesterday_pick = execute_query(yesterday_pick_query, (yesterday_cst,), fetch_one=True, database='ncaamb')

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
    """
    record_result = execute_query(record_query, fetch_one=True, database='ncaamb')

    if record_result and record_result['total_picks']:
        total_picks = int(record_result['total_picks']) if record_result['total_picks'] else 0
        correct_picks = int(record_result['correct_picks']) if record_result['correct_picks'] else 0
        pick_accuracy = (correct_picks / total_picks * 100) if total_picks > 0 else 0.0
    else:
        total_picks = 0
        correct_picks = 0
        pick_accuracy = 0.0

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
    """
    roi_data = execute_query(roi_query, database='ncaamb')

    def american_to_decimal(odds):
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    def decimal_to_american(decimal):
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))

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

    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0
    avg_decimal_odds = sum(decimal_odds_list) / len(decimal_odds_list) if decimal_odds_list else 1.0
    avg_american_odds = decimal_to_american(avg_decimal_odds) if decimal_odds_list else 0

    def format_pick(pick_data):
        if not pick_data:
            return None

        picked_team = pick_data['team_1'] if pick_data['gbm_prob_team_1'] > pick_data['gbm_prob_team_2'] else pick_data['team_2']
        picked_odds = int(pick_data['best_book_odds_team_1']) if pick_data['gbm_prob_team_1'] > pick_data['gbm_prob_team_2'] else int(pick_data['best_book_odds_team_2'])
        matchup = f"{pick_data['team_1']} vs {pick_data['team_2']}"

        result = None
        if pick_data['winning_team']:
            won = (pick_data['gbm_prob_team_1'] > pick_data['gbm_prob_team_2'] and pick_data['winning_team'] == pick_data['team_1']) or \
                  (pick_data['gbm_prob_team_1'] <= pick_data['gbm_prob_team_2'] and pick_data['winning_team'] == pick_data['team_2'])
            result = 'W' if won else 'L'

        return {
            'game_id': pick_data['game_id'],
            'matchup': matchup,
            'picked_team': picked_team,
            'picked_odds': picked_odds,
            'betting_rule': pick_data['betting_rule'],
            'date': str(pick_data['date']),
            'time': str(pick_data['time']) if pick_data['time'] else None,
            'result': result
        }

    return {
        'today_pick': format_pick(today_pick),
        'yesterday_pick': format_pick(yesterday_pick),
        'record': {
            'correct': correct_picks,
            'total': total_picks,
            'accuracy': round(pick_accuracy, 2),
            'roi': round(roi, 2),
            'avg_odds': avg_american_odds
        }
    }
