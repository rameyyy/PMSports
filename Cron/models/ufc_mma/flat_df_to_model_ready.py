import polars as pl
import numpy as np
from datetime import datetime, date

def calculate_fighter_stats(prior_fights, current_fight_date):
    """
    Calculate all statistics for a fighter from their prior fights.
    
    Parameters:
    -----------
    prior_fights : list
        List of prior fight dictionaries
    current_fight_date : date
        Date of the current fight
        
    Returns:
    --------
    dict
        Dictionary containing all calculated statistics
    """
    stats = {}
    
    if not prior_fights or len(prior_fights) == 0:
        # Return defaults if no prior fights
        return get_default_stats()
    
    # Sort fights by date (most recent first)
    sorted_fights = sorted(prior_fights, key=lambda x: x['fight_date'], reverse=True)
    
    # ==================== WIN/LOSS RECORD ====================
    wins = sum(1 for f in sorted_fights if f['result'] == 'win')
    losses = sum(1 for f in sorted_fights if f['result'] == 'loss')
    total = len(sorted_fights)
    
    stats['wins'] = wins
    stats['losses'] = losses
    stats['overall_winrate'] = wins / total if total > 0 else 0
    
    # ==================== RECENT WIN RATES ====================
    # Use available fights, up to requested amount
    last3 = sorted_fights[:min(3, len(sorted_fights))]
    last5 = sorted_fights[:min(5, len(sorted_fights))]
    last7 = sorted_fights[:min(7, len(sorted_fights))]
    
    stats['winrate_last3'] = sum(1 for f in last3 if f['result'] == 'win') / len(last3) if len(last3) > 0 else 0
    stats['winrate_last5'] = sum(1 for f in last5 if f['result'] == 'win') / len(last5) if len(last5) > 0 else 0
    
    # ==================== RECENCY ====================
    # Average days since last N fights (use available fights)
    if len(last3) > 0:
        recency_3 = [(current_fight_date - f['fight_date']).days for f in last3]
        stats['recency_avg_last3'] = np.mean(recency_3)
    else:
        stats['recency_avg_last3'] = 0
    
    if len(last5) > 0:
        recency_5 = [(current_fight_date - f['fight_date']).days for f in last5]
        stats['recency_avg_last5'] = np.mean(recency_5)
    else:
        stats['recency_avg_last5'] = 0
    
    # ==================== WIN STREAK ====================
    win_streak = 0
    for f in sorted_fights:
        if f['result'] == 'win':
            win_streak += 1
        else:
            break
    stats['win_streak'] = win_streak
    
    # ==================== PERFORMANCE STATS ====================
    # Define stat categories with their attempts and landed fields
    stat_fields = [
        ('body', 'body_attempts', 'body_landed'),
        ('clinch', 'clinch_attempts', 'clinch_landed'),
        ('distance', 'distance_attempts', 'distance_landed'),
        ('ground', 'ground_attempts', 'ground_landed'),
        ('head', 'head_attempts', 'head_landed'),
        ('leg', 'leg_attempts', 'leg_landed'),
        ('td', 'td_attempts', 'td_landed'),
        ('total_str', 'total_str_attempts', 'total_str_landed'),
        ('sig_str', 'sig_str_attempts', 'sig_str_landed'),
    ]
    
    for stat_name, attempts_field, landed_field in stat_fields:
        # Last 3 fights (use available)
        stats[f'{stat_name}_attempts_avg_last3'] = calculate_avg(last3, attempts_field)
        stats[f'{stat_name}_landed_avg_last3'] = calculate_avg(last3, landed_field)
        stats[f'{stat_name}_accuracy_avg_last3'] = calculate_accuracy_avg(last3, attempts_field, landed_field)
        
        # Last 5 fights (use available)
        stats[f'{stat_name}_attempts_avg_last5'] = calculate_avg(last5, attempts_field)
        stats[f'{stat_name}_landed_avg_last5'] = calculate_avg(last5, landed_field)
        stats[f'{stat_name}_accuracy_avg_last5'] = calculate_accuracy_avg(last5, attempts_field, landed_field)
        
        # Overall
        stats[f'{stat_name}_attempts_avg_overall'] = calculate_avg(sorted_fights, attempts_field)
        stats[f'{stat_name}_landed_avg_overall'] = calculate_avg(sorted_fights, landed_field)
        stats[f'{stat_name}_accuracy_avg_overall'] = calculate_accuracy_avg(sorted_fights, attempts_field, landed_field)
    
    # ==================== KNOCKDOWNS ====================
    stats['kd_avg_last3'] = calculate_avg(last3, 'kd')
    stats['kd_avg_last5'] = calculate_avg(last5, 'kd')
    stats['kd_avg_overall'] = calculate_avg(sorted_fights, 'kd')
    
    # ==================== SUBMISSIONS ====================
    stats['sub_att_avg_last3'] = calculate_avg(last3, 'sub_att')
    stats['sub_att_avg_last5'] = calculate_avg(last5, 'sub_att')
    stats['sub_att_avg_overall'] = calculate_avg(sorted_fights, 'sub_att')
    
    # ==================== ADVANCED WIN RATE ====================
    # Use available fights for each window
    stats['winrate_advanced_last3'] = calculate_advanced_winrate(last3)
    stats['winrate_advanced_last5'] = calculate_advanced_winrate(last5)
    stats['winrate_advanced_last7'] = calculate_advanced_winrate(last7)
    stats['winrate_advanced_overall'] = calculate_advanced_winrate(sorted_fights)
    
    return stats


def calculate_avg(fights, field):
    """Calculate average of a field across fights."""
    if not fights or len(fights) == 0:
        return 0  # Return 0 instead of None
    # Filter out None values
    values = [f[field] for f in fights if field in f and f[field] is not None]
    return np.mean(values) if values else 0


def calculate_accuracy_avg(fights, attempts_field, landed_field):
    """Calculate average accuracy across fights."""
    if not fights or len(fights) == 0:
        return 0  # Return 0 instead of None
    
    accuracies = []
    for f in fights:
        attempts = f.get(attempts_field, 0)
        landed = f.get(landed_field, 0)
        # Handle None values
        if attempts is None:
            attempts = 0
        if landed is None:
            landed = 0
        if attempts > 0:
            accuracies.append(landed / attempts)
    
    return np.mean(accuracies) if accuracies else 0


def calculate_advanced_winrate(fights):
    """
    Calculate performance-weighted win rate score.
    
    Scoring system:
    - Win by KO/TKO fast = high positive score
    - Win by submission = moderate positive score
    - Win by decision = lower positive score
    - Losses = negative scores (worse if finished early)
    """
    if not fights or len(fights) == 0:
        return 0  # Return 0 instead of None
    
    score = 0
    
    for fight in fights:
        result = fight['result']
        method = fight.get('method', '')
        end_time = fight.get('end_time', '0:00')
        fight_format = fight.get('fight_format', 3)
        
        # Handle None values
        if method is None:
            method = ''
        if end_time is None:
            end_time = '0:00'
        if fight_format is None:
            fight_format = 3
        
        # Convert end_time (MM:SS) to decimal minutes
        try:
            parts = str(end_time).split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            time_decimal = minutes + seconds / 60.0
        except:
            time_decimal = fight_format * 5  # Default to full fight
        
        # Calculate time ratio (how much of fight completed)
        max_time = fight_format * 5
        time_ratio = time_decimal / max_time if max_time > 0 else 1.0
        
        if result == 'win':
            # Base score for win
            if 'ko' in method.lower() or 'tko' in method.lower():
                # KO/TKO: 10 points base, bonus for early finish
                base_score = 10
                time_bonus = (1 - time_ratio) * 5  # Up to 5 bonus points for fast finish
                score += base_score + time_bonus
            elif 'sub' in method.lower():
                # Submission: 8 points base, bonus for early finish
                base_score = 8
                time_bonus = (1 - time_ratio) * 4
                score += base_score + time_bonus
            elif 'd_unan' in method.lower():
                # Unanimous decision: 6 points
                score += 6
            elif 'd_split' in method.lower() or 'd_maj' in method.lower():
                # Split/majority decision: 4.5 points
                score += 4.5
            else:
                # Other wins: 4 points
                score += 4
        else:  # loss
            # Penalties for losses
            if 'ko' in method.lower() or 'tko' in method.lower():
                # KO/TKO loss: -10 base, worse if early
                base_penalty = -10
                time_penalty = (1 - time_ratio) * -5
                score += base_penalty + time_penalty
            elif 'sub' in method.lower():
                # Submission loss: -8 base, worse if early
                base_penalty = -8
                time_penalty = (1 - time_ratio) * -4
                score += base_penalty + time_penalty
            elif 'd_unan' in method.lower():
                # Unanimous decision loss: -6
                score += -6
            elif 'd_split' in method.lower() or 'd_maj' in method.lower():
                # Split/majority decision loss: -4.5
                score += -4.5
            else:
                # Other losses: -4
                score += -4
    
    return score


def get_default_stats():
    """Return default stats for fighters with no prior fights."""
    stats = {
        'wins': 0,
        'losses': 0,
        'overall_winrate': 0,
        'winrate_last3': 0,
        'winrate_last5': 0,
        'recency_avg_last3': 0,
        'recency_avg_last5': 0,
        'win_streak': 0,
    }
    
    # Add all performance stats as 0
    stat_fields = ['body', 'clinch', 'distance', 'ground', 'head', 'leg', 'td', 'total_str', 'sig_str']
    for stat_name in stat_fields:
        for window in ['last3', 'last5', 'overall']:
            stats[f'{stat_name}_attempts_avg_{window}'] = 0
            stats[f'{stat_name}_landed_avg_{window}'] = 0
            stats[f'{stat_name}_accuracy_avg_{window}'] = 0
    
    stats['kd_avg_last3'] = 0
    stats['kd_avg_last5'] = 0
    stats['kd_avg_overall'] = 0
    
    stats['sub_att_avg_last3'] = 0
    stats['sub_att_avg_last5'] = 0
    stats['sub_att_avg_overall'] = 0
    
    stats['winrate_advanced_last3'] = 0
    stats['winrate_advanced_last5'] = 0
    stats['winrate_advanced_last7'] = 0
    stats['winrate_advanced_overall'] = 0
    
    return stats


def create_differential_features(df):
    """
    Transform fight snapshots into differential features for ML model.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Polars DataFrame with fight snapshots including prior_f1 and prior_f2 lists
    
    Returns:
    --------
    pl.DataFrame
        Transformed dataframe with differential features and target variable
    """
    
    result_rows = []
    
    # Convert to Python for easier processing of nested lists
    df_python = df.to_dicts()
    
    for row in df_python:
        features = {}
        
        # ==================== METADATA (for train/test splitting) ====================
        features['fight_date'] = row['fight_date']
        features['fight_id'] = row['fight_id']
        
        # ==================== BASIC DIFFERENTIALS ====================
        features['height_diff'] = row['f1_height_in'] - row['f2_height_in']
        features['reach_diff'] = row['f1_reach_in'] - row['f2_reach_in']
        
        # Age difference in days at fight time
        f1_age_days = (row['fight_date'] - row['f1_dob']).days
        f2_age_days = (row['fight_date'] - row['f2_dob']).days
        features['age_diff'] = f1_age_days - f2_age_days
        
        features['experience_diff'] = row['prior_cnt_f1'] - row['prior_cnt_f2']
        
        # ==================== CATEGORICAL FEATURES ====================
        features['weight_class'] = row['weight_class']
        features['stance_matchup'] = f"{row['f1_stance']}_vs_{row['f2_stance']}"
        
        # ==================== EXTRACT PRIOR FIGHT STATS ====================
        prior_f1 = row['prior_f1'] if row['prior_f1'] is not None else []
        prior_f2 = row['prior_f2'] if row['prior_f2'] is not None else []
        
        # Calculate stats for both fighters
        f1_stats = calculate_fighter_stats(prior_f1, row['fight_date'])
        f2_stats = calculate_fighter_stats(prior_f2, row['fight_date'])
        
        # ==================== RECORD DIFFERENTIALS ====================
        features['wins_diff'] = f1_stats['wins'] - f2_stats['wins']
        features['losses_diff'] = f1_stats['losses'] - f2_stats['losses']
        features['overall_winrate_diff'] = f1_stats['overall_winrate'] - f2_stats['overall_winrate']
        
        # ==================== RECENT FORM DIFFERENTIALS ====================
        features['winrate_last3_diff'] = f1_stats['winrate_last3'] - f2_stats['winrate_last3']
        features['winrate_last5_diff'] = f1_stats['winrate_last5'] - f2_stats['winrate_last5']
        
        # ==================== RECENCY DIFFERENTIALS ====================
        features['recency_avg_last3_diff'] = f1_stats['recency_avg_last3'] - f2_stats['recency_avg_last3']
        features['recency_avg_last5_diff'] = f1_stats['recency_avg_last5'] - f2_stats['recency_avg_last5']
        
        # ==================== WIN STREAK ====================
        features['win_streak_diff'] = f1_stats['win_streak'] - f2_stats['win_streak']
        
        # ==================== PERFORMANCE STAT DIFFERENTIALS ====================
        stat_fields = ['body', 'clinch', 'distance', 'ground', 'head', 'leg', 'td', 'total_str', 'sig_str']
        
        for stat_name in stat_fields:
            for window in ['last3', 'last5', 'overall']:
                for metric in ['attempts_avg', 'landed_avg', 'accuracy_avg']:
                    key = f'{stat_name}_{metric}_{window}'
                    features[f'{key}_diff'] = f1_stats[key] - f2_stats[key]
        
        # ==================== KNOCKDOWN DIFFERENTIALS ====================
        features['kd_avg_last3_diff'] = f1_stats['kd_avg_last3'] - f2_stats['kd_avg_last3']
        features['kd_avg_last5_diff'] = f1_stats['kd_avg_last5'] - f2_stats['kd_avg_last5']
        features['kd_avg_overall_diff'] = f1_stats['kd_avg_overall'] - f2_stats['kd_avg_overall']
        
        # ==================== SUBMISSION DIFFERENTIALS ====================
        features['sub_att_avg_last3_diff'] = f1_stats['sub_att_avg_last3'] - f2_stats['sub_att_avg_last3']
        features['sub_att_avg_last5_diff'] = f1_stats['sub_att_avg_last5'] - f2_stats['sub_att_avg_last5']
        features['sub_att_avg_overall_diff'] = f1_stats['sub_att_avg_overall'] - f2_stats['sub_att_avg_overall']
        
        # ==================== ADVANCED WIN RATE DIFFERENTIALS ====================
        features['winrate_advanced_last3_diff'] = f1_stats['winrate_advanced_last3'] - f2_stats['winrate_advanced_last3']
        features['winrate_advanced_last5_diff'] = f1_stats['winrate_advanced_last5'] - f2_stats['winrate_advanced_last5']
        features['winrate_advanced_last7_diff'] = f1_stats['winrate_advanced_last7'] - f2_stats['winrate_advanced_last7']
        features['winrate_advanced_overall_diff'] = f1_stats['winrate_advanced_overall'] - f2_stats['winrate_advanced_overall']
        
        # ==================== TARGET VARIABLE ====================
        features['target'] = 1 if row['winner_id'] == row['fighter1_id'] else 0
        
        result_rows.append(features)
    
    # Convert to Polars DataFrame
    result_df = pl.DataFrame(result_rows)
    
    return result_df


def safe_subtract(val1, val2):
    """Safely subtract two values, handling None cases."""
    if val1 is None or val2 is None:
        return None
    return val1 - val2


# Example usage:
# differential_df = create_differential_features(fight_snapshots_df)
# print(differential_df.head())
# print(f"Shape: {differential_df.shape}")