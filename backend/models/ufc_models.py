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
    """Get past UFC events"""
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
    """Get all fights for a specific event with fighter details and predictions"""
    query = """
        SELECT 
            f.fight_id,
            f.event_id,
            f.fighter1_id,
            f.fighter2_id,
            f.winner_id,
            f.loser_id,
            f.fighter1_name,
            f.fighter2_name,
            f.fight_date,
            f.fight_link,
            f.method,
            f.fight_format,
            f.fight_type,
            f.referee,
            f.end_time,
            f.weight_class,
            f1.nickname as fighter1_nickname,
            f1.img_link as fighter1_img,
            f2.nickname as fighter2_nickname,
            f2.img_link as fighter2_img,
            p.logistic_pred,
            p.logistic_f1_prob,
            p.xgboost_pred,
            p.xgboost_f1_prob,
            p.gradient_pred,
            p.gradient_f1_prob,
            p.homemade_pred,
            p.homemade_f1_prob,
            p.prediction_confidence,
            p.predicted_winner,
            p.logistic_correct,
            p.xgboost_correct,
            p.gradient_correct,
            p.homemade_correct
        FROM ufc.fights f
        LEFT JOIN ufc.fighters f1 ON f.fighter1_id = f1.fighter_id
        LEFT JOIN ufc.fighters f2 ON f.fighter2_id = f2.fighter_id
        LEFT JOIN ufc.predictions p ON f.fight_id = p.fight_id
        WHERE f.event_id = %s
        ORDER BY f.fight_date DESC
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
            vigor
        FROM ufc.bookmaker_odds
        WHERE fight_id = %s
    """
    return execute_query(query, (fight_id,))

def get_model_stats():
    """Get overall model performance statistics for each model type"""
    query = """
        SELECT 
            'Logistic Regression' as model_name,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN logistic_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(
                (SUM(CASE WHEN logistic_correct = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 
                2
            ) as accuracy_percentage
        FROM ufc.predictions
        WHERE logistic_correct IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'XGBoost' as model_name,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN xgboost_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(
                (SUM(CASE WHEN xgboost_correct = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 
                2
            ) as accuracy_percentage
        FROM ufc.predictions
        WHERE xgboost_correct IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'Gradient Boosting' as model_name,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN gradient_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(
                (SUM(CASE WHEN gradient_correct = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 
                2
            ) as accuracy_percentage
        FROM ufc.predictions
        WHERE gradient_correct IS NOT NULL
        
        UNION ALL
        
        SELECT 
            'Homemade Model' as model_name,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN homemade_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(
                (SUM(CASE WHEN homemade_correct = 1 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 
                2
            ) as accuracy_percentage
        FROM ufc.predictions
        WHERE homemade_correct IS NOT NULL
    """
    return execute_query(query)

def get_calibrated_probability(model_name, raw_probability):
    """
    Get calibrated probability based on historical accuracy at that confidence level
    Uses ±1% range around the raw probability
    """
    # Map model names to database columns
    prob_column_map = {
        'logistic': 'logistic_f1_prob',
        'xgboost': 'xgboost_f1_prob',
        'gradient': 'gradient_f1_prob',
        'homemade': 'homemade_f1_prob'
    }
    
    pred_column_map = {
        'logistic': 'logistic_pred',
        'xgboost': 'xgboost_pred',
        'gradient': 'gradient_pred',
        'homemade': 'homemade_pred'
    }
    
    prob_col = prob_column_map.get(model_name.lower())
    pred_col = pred_column_map.get(model_name.lower())
    
    if not prob_col or not pred_col:
        return raw_probability
    
    # Get predictions within ±1% of the raw probability
    lower_bound = raw_probability - 0.01
    upper_bound = raw_probability + 0.01
    
    query = f"""
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE 
                WHEN ({pred_col} = 1 AND actual_winner = 1) OR ({pred_col} = 0 AND actual_winner = 0)
                THEN 1 ELSE 0 
            END) as correct_predictions,
            ROUND(
                (SUM(CASE 
                    WHEN ({pred_col} = 1 AND actual_winner = 1) OR ({pred_col} = 0 AND actual_winner = 0)
                    THEN 1 ELSE 0 
                END) / COUNT(*)) * 100, 
                1
            ) as calibrated_probability
        FROM ufc.predictions
        WHERE {prob_col} >= %s 
        AND {prob_col} <= %s
        AND actual_winner IS NOT NULL
    """
    
    result = execute_query(query, (lower_bound, upper_bound), fetch_one=True)
    
    if result and result['total_predictions'] >= 5:  # Need at least 5 samples
        return result['calibrated_probability']
    else:
        return round(raw_probability * 100, 1)  # Fall back to raw if not enough data

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