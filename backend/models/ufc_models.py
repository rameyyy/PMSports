from utils.db import execute_query
from datetime import datetime

def get_all_events():
    """Get all UFC events"""
    query = """
        SELECT 
            event_id,
            event_url,
            title,
            event_datestr,
            location,
            date
        FROM ufc.events
        ORDER BY date DESC
    """
    return execute_query(query)

def get_model_accuracies():
    """Get model accuracy statistics from the model_accuracies table"""
    query = """
        SELECT 
            model_name,
            total_predictions,
            correct_predictions,
            accuracy,
            avg_confidence,
            avg_sample_size
        FROM ufc.model_accuracies
        ORDER BY accuracy DESC
    """
    return execute_query(query)

def get_all_bets(offset=0, limit=10):
    """Get all user bets paginated by event (not individual bets)"""
    # First, get distinct events with their earliest fight_date for ordering
    events_query = """
        SELECT DISTINCT event_name, fight_date
        FROM ufc.bets
        ORDER BY fight_date DESC
        LIMIT %s OFFSET %s
    """
    events = execute_query(events_query, (limit, offset))

    if not events:
        return []

    # Build list of event identifiers
    event_list = [(event['event_name'], event['fight_date']) for event in events]

    # Now fetch all bets for these events
    placeholders = ','.join(['(%s, %s)'] * len(event_list))
    params = []
    for event_name, fight_date in event_list:
        params.extend([event_name, fight_date])

    bets_query = f"""
        SELECT
            bet_date,
            bet_outcome,
            bet_type,
            event_name,
            fight_date,
            fighter1_name,
            fighter2_name,
            fighter1_odds,
            fighter2_odds,
            fighter1_ev,
            fighter2_ev,
            fighter1_pred,
            fighter2_pred,
            fighter_bet_on,
            sportsbook,
            stake,
            potential_profit,
            potential_loss
        FROM ufc.bets
        WHERE (event_name, fight_date) IN ({placeholders})
        ORDER BY fight_date DESC, bet_date DESC
    """
    return execute_query(bets_query, tuple(params))

def get_all_bets_total_count():
    """Get total count of distinct events for pagination"""
    query = """
        SELECT COUNT(DISTINCT event_name, fight_date) as total
        FROM ufc.bets
    """
    result = execute_query(query, fetch_one=True)
    return result['total'] if result else 0

def get_pending_bets(offset=0, limit=10):
    """Get pending bets paginated by event (not individual bets)"""
    # First, get distinct events with pending bets
    events_query = """
        SELECT DISTINCT event_name, fight_date
        FROM ufc.bets
        WHERE bet_outcome = 'pending'
        ORDER BY fight_date ASC
        LIMIT %s OFFSET %s
    """
    events = execute_query(events_query, (limit, offset))

    if not events:
        return []

    # Build list of event identifiers
    event_list = [(event['event_name'], event['fight_date']) for event in events]

    # Now fetch all pending bets for these events
    placeholders = ','.join(['(%s, %s)'] * len(event_list))
    params = []
    for event_name, fight_date in event_list:
        params.extend([event_name, fight_date])

    bets_query = f"""
        SELECT
            bet_date,
            bet_outcome,
            bet_type,
            event_name,
            fight_date,
            fighter1_name,
            fighter2_name,
            fighter1_odds,
            fighter2_odds,
            fighter1_ev,
            fighter2_ev,
            fighter1_pred,
            fighter2_pred,
            fighter_bet_on,
            sportsbook,
            stake,
            potential_profit,
            potential_loss
        FROM ufc.bets
        WHERE bet_outcome = 'pending' AND (event_name, fight_date) IN ({placeholders})
        ORDER BY fight_date ASC, bet_date DESC
    """
    return execute_query(bets_query, tuple(params))

def get_pending_bets_total_count():
    """Get total count of distinct events with pending bets for pagination"""
    query = """
        SELECT COUNT(DISTINCT event_name, fight_date) as total
        FROM ufc.bets
        WHERE bet_outcome = 'pending'
    """
    result = execute_query(query, fetch_one=True)
    return result['total'] if result else 0

def get_settled_bets(offset=0, limit=10):
    """Get settled bets (won/lost) paginated by event (not individual bets)"""
    # First, get distinct events with settled bets
    events_query = """
        SELECT DISTINCT event_name, fight_date
        FROM ufc.bets
        WHERE bet_outcome IN ('won', 'lost', 'push')
        ORDER BY fight_date DESC
        LIMIT %s OFFSET %s
    """
    events = execute_query(events_query, (limit, offset))

    if not events:
        return []

    # Build list of event identifiers
    event_list = [(event['event_name'], event['fight_date']) for event in events]

    # Now fetch all settled bets for these events
    placeholders = ','.join(['(%s, %s)'] * len(event_list))
    params = []
    for event_name, fight_date in event_list:
        params.extend([event_name, fight_date])

    bets_query = f"""
        SELECT
            bet_date,
            bet_outcome,
            bet_type,
            event_name,
            fight_date,
            fighter1_name,
            fighter2_name,
            fighter1_odds,
            fighter2_odds,
            fighter1_ev,
            fighter2_ev,
            fighter1_pred,
            fighter2_pred,
            fighter_bet_on,
            sportsbook,
            stake,
            potential_profit,
            potential_loss
        FROM ufc.bets
        WHERE bet_outcome IN ('won', 'lost', 'push') AND (event_name, fight_date) IN ({placeholders})
        ORDER BY fight_date DESC, bet_date DESC
    """
    return execute_query(bets_query, tuple(params))

def get_settled_bets_total_count():
    """Get total count of distinct events with settled bets for pagination"""
    query = """
        SELECT COUNT(DISTINCT event_name, fight_date) as total
        FROM ufc.bets
        WHERE bet_outcome IN ('won', 'lost', 'push')
    """
    result = execute_query(query, fetch_one=True)
    return result['total'] if result else 0

def get_betting_stats():
    """Get overall betting statistics"""
    query = """
        SELECT 
            COUNT(*) as total_bets,
            SUM(stake) as total_staked,
            SUM(CASE WHEN bet_outcome = 'won' THEN potential_profit ELSE 0 END) as total_profit,
            SUM(CASE WHEN bet_outcome = 'lost' THEN potential_loss ELSE 0 END) as total_loss,
            COUNT(CASE WHEN bet_outcome = 'won' THEN 1 END) as bets_won,
            COUNT(CASE WHEN bet_outcome = 'lost' THEN 1 END) as bets_lost,
            COUNT(CASE WHEN bet_outcome = 'pending' THEN 1 END) as bets_pending,
            ROUND(
                (COUNT(CASE WHEN bet_outcome = 'won' THEN 1 END) * 100.0 / 
                NULLIF(COUNT(CASE WHEN bet_outcome IN ('won', 'lost') THEN 1 END), 0)), 
                2
            ) as win_rate,
            ROUND(
                ((SUM(CASE WHEN bet_outcome = 'won' THEN potential_profit ELSE 0 END) - 
                  SUM(CASE WHEN bet_outcome = 'lost' THEN potential_loss ELSE 0 END)) * 100.0 / 
                NULLIF(SUM(CASE WHEN bet_outcome IN ('won', 'lost') THEN stake END), 0)),
                2
            ) as roi
        FROM ufc.bets
    """
    return execute_query(query, fetch_one=True)

def get_upcoming_events():
    """Get upcoming UFC events"""
    query = """
        SELECT 
            event_id,
            event_url,
            title,
            event_datestr,
            location,
            date
        FROM ufc.events
        WHERE date >= CURDATE()
        ORDER BY date ASC
    """
    return execute_query(query)

def get_past_events(offset=0, limit=10):
    """Get past UFC events with pagination (only events after 2025-10-03)"""
    query = """
        SELECT
            event_id,
            event_url,
            title,
            event_datestr,
            location,
            date
        FROM ufc.events
        WHERE date < CURDATE()
          AND date > '2025-10-03'
        ORDER BY date DESC
        LIMIT %s OFFSET %s
    """
    return execute_query(query, (limit, offset))

def get_past_events_total_count():
    """Get total count of past UFC events for pagination"""
    query = """
        SELECT COUNT(*) as total
        FROM ufc.events
        WHERE date < CURDATE()
          AND date > '2025-10-03'
    """
    result = execute_query(query, fetch_one=True)
    return result['total'] if result else 0

def get_event_by_id(event_id):
    """Get a specific event by ID"""
    query = """
        SELECT 
            event_id,
            event_url,
            title,
            event_datestr,
            location,
            date
        FROM ufc.events
        WHERE event_id = %s
    """
    return execute_query(query, (event_id,), fetch_one=True)

def get_fights_by_event(event_id):
    """Get all fights for a specific event from prediction_simplified"""
    query = """
        SELECT 
            fight_id,
            event_id,
            fighter1_id,
            fighter2_id,
            fighter1_name,
            fighter2_name,
            fighter1_nickname,
            fighter2_nickname,
            fighter1_img_link,
            fighter2_img_link,
            algopick_model,
            algopick_prediction,
            algopick_probability,
            window_sample,
            correct,
            date,
            end_time,
            weight_class,
            fight_type,
            win_method
        FROM ufc.prediction_simplified
        WHERE event_id = %s
        ORDER BY date DESC
    """
    return execute_query(query, (event_id,))

def get_fight_odds(fight_id):
    """Get bookmaker odds for a specific fight with EV calculations"""
    query = """
        SELECT 
            bo.fight_id,
            bo.bookmaker,
            bo.fighter1_id,
            bo.fighter2_id,
            bo.fighter1_odds,
            bo.fighter2_odds,
            bo.fighter1_odds_percent,
            bo.fighter2_odds_percent,
            bo.fighter1_ev,
            bo.vigor,
            bo.fighter2_ev,
            ps.algopick_prediction,
            ps.algopick_probability
        FROM ufc.bookmaker_odds bo
        LEFT JOIN ufc.prediction_simplified ps 
            ON bo.fight_id = ps.fight_id
        WHERE bo.fight_id = %s
    """
    return execute_query(query, (fight_id,))

# def get_model_stats():
#     """Get overall model performance statistics from prediction_simplified (post Oct 3, 2025)"""
#     query = """
#         SELECT 
#             algopick_model as model_name,
#             COUNT(*) as total_predictions,
#             SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
#             ROUND(
#                 (SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 
#                 2
#             ) as accuracy_percentage,
#             ROUND(AVG(algopick_probability), 2) as avg_confidence,
#             ROUND(AVG(window_sample), 0) as avg_sample_size
#         FROM ufc.prediction_simplified
#         WHERE correct IS NOT NULL
#           AND date > '2025-10-03'
#         GROUP BY algopick_model
#     """
#     return execute_query(query)

def get_fight_prediction(fight_id):
    """Get the simplified prediction for a specific fight (post Oct 3, 2025)"""
    query = """
        SELECT 
            fight_id,
            fighter1_name,
            fighter2_name,
            fighter1_nickname,
            fighter2_nickname,
            fighter1_img_link,
            fighter2_img_link,
            algopick_model,
            algopick_prediction,
            algopick_probability,
            window_sample,
            correct,
            date,
            end_time,
            weight_class,
            fight_type,
            win_method
        FROM ufc.prediction_simplified
        WHERE fight_id = %s
          AND date > '2025-10-03'
    """
    return execute_query(query, (fight_id,), fetch_one=True)

def get_recent_predictions(limit=10):
    """Get most recent predictions (post Oct 3, 2025)"""
    query = """
        SELECT 
            fight_id,
            event_id,
            fighter1_name,
            fighter2_name,
            fighter1_img_link,
            fighter2_img_link,
            algopick_model,
            algopick_prediction,
            algopick_probability,
            window_sample,
            correct,
            date,
            weight_class,
            fight_type
        FROM ufc.prediction_simplified
        WHERE date > '2025-10-03'
        ORDER BY date DESC
        LIMIT %s
    """
    return execute_query(query, (limit,))

def get_predictions_by_confidence(min_confidence=60.0, limit=20):
    """Get predictions above a certain confidence threshold (post Oct 3, 2025)"""
    query = """
        SELECT 
            fight_id,
            event_id,
            fighter1_name,
            fighter2_name,
            fighter1_img_link,
            fighter2_img_link,
            algopick_model,
            algopick_prediction,
            algopick_probability,
            window_sample,
            correct,
            date,
            weight_class,
            fight_type
        FROM ufc.prediction_simplified
        WHERE algopick_probability >= %s
          AND date > '2025-10-03'
        ORDER BY algopick_probability DESC, date DESC
        LIMIT %s
    """
    return execute_query(query, (min_confidence, limit))

def get_recent_predictions(limit=10):
    """Get most recent predictions (post Oct 3, 2025)"""
    query = """
        SELECT 
            fight_id,
            event_id,
            fighter1_name,
            fighter2_name,
            fighter1_img_link,
            fighter2_img_link,
            algopick_model,
            algopick_prediction,
            algopick_probability,
            window_sample,
            correct,
            date,
            weight_class
        FROM ufc.prediction_simplified
        WHERE date > '2025-10-03'
        ORDER BY date DESC
        LIMIT %s
    """
    return execute_query(query, (limit,))

def get_predictions_by_confidence(min_confidence=60.0, limit=20):
    """Get predictions above a certain confidence threshold (post Oct 3, 2025)"""
    query = """
        SELECT 
            fight_id,
            event_id,
            fighter1_name,
            fighter2_name,
            fighter1_img_link,
            fighter2_img_link,
            algopick_model,
            algopick_prediction,
            algopick_probability,
            window_sample,
            correct,
            date,
            weight_class
        FROM ufc.prediction_simplified
        WHERE algopick_probability >= %s
          AND date > '2025-10-03'
        ORDER BY algopick_probability DESC, date DESC
        LIMIT %s
    """
    return execute_query(query, (min_confidence, limit))

def get_fighter_stats(fighter_id):
    """Get fighter statistics"""
    query = """
        SELECT 
            fighter_id,
            name,
            nickname,
            img_link,
            height_in,
            weight_lbs,
            reach_in,
            stance,
            dob,
            slpm,
            str_acc,
            sapm,
            str_def,
            td_avg,
            td_acc,
            td_def,
            sub_avg,
            win,
            loss,
            draw
        FROM ufc.fighters
        WHERE fighter_id = %s
    """
    return execute_query(query, (fighter_id,), fetch_one=True)