"""
NCAA Men's Basketball (ncaamb) data models and database queries
"""
from utils.db import execute_query


def get_homepage_stats():
    """Get homepage stats from pre-computed table"""
    from datetime import datetime
    import pytz

    cst = pytz.timezone('America/Chicago')
    current_time_cst = datetime.now(cst)
    today_cst = current_time_cst.strftime('%Y-%m-%d')

    # Check for blackout period (before 8:15am CST)
    hour = current_time_cst.hour
    minute = current_time_cst.minute
    is_blackout = (hour < 8) or (hour == 8 and minute < 15)

    query = """
        SELECT * FROM ncaamb.homepage_stats
        WHERE date = %s
    """
    result = execute_query(query, (today_cst,), fetch_one=True, database='ncaamb')

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
