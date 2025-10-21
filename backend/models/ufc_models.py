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

def get_past_events():
    """Get past UFC events (only events after 2025-10-03)"""
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
    """
    return execute_query(query)

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
    """Get bookmaker odds for a specific fight"""
    query = """
        SELECT 
            fight_id,
            bookmaker,
            fighter1_id,
            fighter2_id,
            fighter1_odds,
            fighter2_odds,
            fighter1_odds_percent,
            fighter2_odds_percent,
            ev,
            vigor
        FROM ufc.bookmaker_odds
        WHERE fight_id = %s
    """
    return execute_query(query, (fight_id,))

def get_model_stats():
    """Get overall model performance statistics from prediction_simplified (post Oct 3, 2025)"""
    query = """
        SELECT 
            algopick_model as model_name,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(
                (SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 
                2
            ) as accuracy_percentage,
            ROUND(AVG(algopick_probability), 2) as avg_confidence,
            ROUND(AVG(window_sample), 0) as avg_sample_size
        FROM ufc.prediction_simplified
        WHERE correct IS NOT NULL
          AND date > '2025-10-03'
        GROUP BY algopick_model
    """
    return execute_query(query)

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