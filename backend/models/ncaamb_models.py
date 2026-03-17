"""
NCAA Men's Basketball (ncaamb) data models and database queries
"""
from decimal import Decimal
from utils.db import execute_query


def get_homepage_stats():
    """Get homepage stats from pre-computed table"""
    from datetime import datetime, timedelta
    import pytz

    cst = pytz.timezone('America/Chicago')
    current_time_cst = datetime.now(cst)
    hour = current_time_cst.hour
    minute = current_time_cst.minute

    # Check for blackout period (before 8:15am CST)
    is_blackout = (hour < 8) or (hour == 8 and minute < 15)

    if hour < 8:
        # 12am-8am: today's data doesn't exist yet, use yesterday
        lookup_date = (current_time_cst - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        lookup_date = current_time_cst.strftime('%Y-%m-%d')

    # Fall back to most recent date with data
    query = """
        SELECT * FROM ncaamb.homepage_stats
        WHERE date = (
            SELECT MAX(date) FROM ncaamb.homepage_stats WHERE date <= %s
        )
    """
    result = execute_query(query, (lookup_date,), fetch_one=True, database='ncaamb')

    if not result:
        return None

    return {
        'date': str(result['date']),
        'todays_games_count': -1 if is_blackout else (result['todays_games_count'] or 0),
        'my_accuracy': float(result['my_accuracy'] or 0),
        'vegas_accuracy': float(result['vegas_accuracy'] or 0),
        'my_total_correct': result['my_total_correct'] or 0,
        'vegas_total_correct': result['vegas_total_correct'] or 0,
        'total_complete_matches': result['total_complete_matches'] or 0,
        'pick_of_day_acc': float(result['pick_of_day_acc'] or 0),
        'pick_of_day_correct': result['pick_of_day_correct'] or 0,
        'pick_of_day_total': result['pick_of_day_total'] or 0,
        'pod_avg_odds': result['pod_avg_odds'] or 0,
        'pod_roi': float(result['pod_roi'] or 0),
        'pod_td_matchup': result['pod_td_matchup'],
        'pod_td_pick': result['pod_td_pick'],
        'pod_td_odds': result['pod_td_odds'],
        'pod_yd_matchup': result['pod_yd_matchup'],
        'pod_yd_pick': result['pod_yd_pick'],
        'pod_yd_odds': result['pod_yd_odds'],
        'pod_yd_outcome': result['pod_yd_outcome']
    }


def get_bracket_predictions(year=2026):
    """Get all bracket predictions for a given year."""
    query = """
        SELECT id, bracket_slot, round, region,
               pred_team_1, pred_team_1_seed, pred_team_2, pred_team_2_seed,
               prob_lgb, prob_xgb, prob_logistic, prob_ensemble,
               predicted_winner, predicted_winner_seed,
               game_id, actual_team_1, actual_team_2, actual_winner, correct
        FROM bracket_predictions
        WHERE bracket_year = %s
        ORDER BY id
    """
    rows = execute_query(query, (year,), database='ncaamb')
    if not rows:
        return []
    result = []
    for r in rows:
        result.append({
            'id': r['id'],
            'bracket_slot': r['bracket_slot'],
            'round': r['round'],
            'region': r['region'],
            'pred_team_1': r['pred_team_1'],
            'pred_team_1_seed': r['pred_team_1_seed'],
            'pred_team_2': r['pred_team_2'],
            'pred_team_2_seed': r['pred_team_2_seed'],
            'prob_lgb': float(r['prob_lgb']) if r['prob_lgb'] is not None else None,
            'prob_xgb': float(r['prob_xgb']) if r['prob_xgb'] is not None else None,
            'prob_logistic': float(r['prob_logistic']) if r['prob_logistic'] is not None else None,
            'prob_ensemble': float(r['prob_ensemble']) if r['prob_ensemble'] is not None else None,
            'predicted_winner': r['predicted_winner'],
            'predicted_winner_seed': r['predicted_winner_seed'],
            'game_id': r['game_id'],
            'actual_team_1': r['actual_team_1'],
            'actual_team_2': r['actual_team_2'],
            'actual_winner': r['actual_winner'],
            'correct': r['correct'],
        })
    return result


def get_games_by_date(date_str):
    """Get all games for a given date from ui_games table"""
    query = """
        SELECT * FROM ncaamb.ui_games
        WHERE date = %s
        ORDER BY id
    """
    results = execute_query(query, (date_str,), database='ncaamb')

    if not results:
        return []

    games = []
    for row in results:
        games.append({
            'game_id': row['game_id'],
            'date': str(row['date']),
            'team_1': row['team_1'],
            'team_1_hna': row['team_1_hna'],
            'team_2': row['team_2'],
            'team_1_rank': row['team_1_rank'],
            'team_2_rank': row['team_2_rank'],
            'team_1_conference': row['team_1_conference'],
            'team_2_conference': row['team_2_conference'],
            'team_1_prob_algopicks': float(row['team_1_prob_algopicks']) if row['team_1_prob_algopicks'] is not None else None,
            'team_2_prob_algopicks': float(row['team_2_prob_algopicks']) if row['team_2_prob_algopicks'] is not None else None,
            'team_1_prob_vegas': float(row['team_1_prob_vegas']) if row['team_1_prob_vegas'] is not None else None,
            'team_2_prob_vegas': float(row['team_2_prob_vegas']) if row['team_2_prob_vegas'] is not None else None,
            'team_1_ml': row['team_1_ml'],
            'team_2_ml': row['team_2_ml'],
            'total_pred': row['total_pred'],
            'vegas_ou_line': float(row['vegas_ou_line']) if row['vegas_ou_line'] is not None else None,
            'actual_total': row['actual_total'],
            'actual_winner': row['actual_winner'],
        })

    return games


def get_model_performance():
    """Get latest model performance stats from pre-computed table"""
    from datetime import datetime, timedelta
    import pytz

    cst = pytz.timezone('America/Chicago')
    now_cst = datetime.now(cst)
    hour = now_cst.hour

    if hour < 8:
        # 12am-8am: data for today doesn't exist yet, use yesterday
        lookup_date = (now_cst - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        # 8am+: try today first
        lookup_date = now_cst.strftime('%Y-%m-%d')

    query = """
        SELECT * FROM ncaamb.model_performance
        WHERE date = (
            SELECT MAX(date) FROM ncaamb.model_performance WHERE date <= %s
        )
        ORDER BY book
    """
    results = execute_query(query, (lookup_date,), database='ncaamb')

    if not results:
        return None

    rows = []
    for r in results:
        rows.append({
            'date': str(r['date']),
            'book': r['book'],
            'ml_right': r['ml_right'],
            'ml_total': r['ml_total'],
            'ou_mae': float(r['ou_mae']) if r['ou_mae'] is not None else None,
            'ou_games': r['ou_games'],
            'ap_ou_right': r['ap_ou_right'],
            'ap_ou_total': r['ap_ou_total'],
            'ap_ou_acc': float(r['ap_ou_acc']) if r['ap_ou_acc'] is not None else None,
            'ap_over_acc': float(r['ap_over_acc']) if r['ap_over_acc'] is not None else None,
            'ap_under_acc': float(r['ap_under_acc']) if r['ap_under_acc'] is not None else None,
        })

    return {'date': rows[0]['date'], 'books': rows}
