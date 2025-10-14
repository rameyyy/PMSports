# extract_fight_features.py

import polars as pl
import numpy as np
from typing import Dict, List
from datetime import date

def extract_features_from_snapshots(df: pl.DataFrame) -> pl.DataFrame:
    """
    Simple wrapper to extract fight features from fight_snapshots dataframe.
    
    Args:
        df: DataFrame with prior_f1 and prior_f2 columns (fight snapshots)
    
    Returns:
        DataFrame with extracted features and differentials
    """
    print(f"Extracting features from {len(df)} fights...")
    
    # Extract features
    features_df = extract_fight_history_features(df)
    
    # Calculate differentials
    features_df = calculate_feature_differentials(features_df)
    
    print(f"✅ Extracted features for {len(features_df)} fights")
    
    return features_df


def extract_fight_history_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract aggregate features from fight history lists in fight_snapshots.parquet.
    Enhanced with RECENCY WEIGHTING - recent fights matter more!
    """
    
    def safe_get(fight_dict, key, default=0):
        """Safely get a value from dict, handling None"""
        val = fight_dict.get(key, default)
        return default if val is None else val
    
    def calc_recency_weights(fights_list, current_date, half_life_days=365):
        """
        Calculate exponential decay weights based on fight age.
        half_life_days: how many days for a fight to lose 50% importance
        Default 365 = fights from 1 year ago are worth 50% of today's fights
        """
        if not fights_list:
            return []
        
        weights = []
        for fight in fights_list:
            fight_date = fight.get('fight_date')
            if fight_date:
                days_ago = (current_date - fight_date).days
                # Exponential decay: weight = 2^(-days_ago / half_life)
                weight = 2 ** (-days_ago / half_life_days)
                weights.append(weight)
            else:
                weights.append(0.5)  # Default weight if no date
        
        return weights
    
    def calc_weighted_avg_stat(fights_list, stat_name, weights):
        """Average of a stat weighted by recency"""
        if not fights_list or not weights or len(fights_list) != len(weights):
            return None
        
        values = [safe_get(f, stat_name, 0) * w for f, w in zip(fights_list, weights)]
        total_weight = sum(weights)
        
        return float(sum(values) / total_weight) if total_weight > 0 else None
    
    def calc_weighted_win_rate(fights_list, weights):
        """Win rate weighted by recency"""
        if not fights_list or not weights:
            return None
        
        weighted_wins = sum(w for f, w in zip(fights_list, weights) if f.get('result') == 'win')
        total_weight = sum(weights)
        
        return float(weighted_wins / total_weight) if total_weight > 0 else None
    
    def calc_weighted_accuracy(fights_list, landed_stat, attempts_stat, weights):
        """Accuracy weighted by recency"""
        if not fights_list or not weights:
            return None
        
        total_landed = sum(safe_get(f, landed_stat, 0) * w for f, w in zip(fights_list, weights))
        total_attempts = sum(safe_get(f, attempts_stat, 0) * w for f, w in zip(fights_list, weights))
        
        return float(total_landed / total_attempts) if total_attempts > 0 else None
    
    def calc_current_streak(fights_list):
        """Calculate current win/loss streak (positive = wins, negative = losses)"""
        if not fights_list or len(fights_list) == 0:
            return 0
        
        streak = 0
        last_result = None
        
        for fight in reversed(fights_list):
            result = fight.get('result')
            if result in ['win', 'loss']:
                if last_result is None:
                    last_result = result
                    streak = 1 if result == 'win' else -1
                elif result == last_result:
                    streak += 1 if result == 'win' else -1
                else:
                    break
        
        return float(streak)
    
    def calc_finish_quality_score(fights_list):
        """
        Score based on how fights were won/lost.
        Finishes (KO/Sub) are worth more than decisions.
        """
        if not fights_list or len(fights_list) == 0:
            return None
        
        score = 0
        for fight in fights_list:
            result = fight.get('result')
            method = safe_get(fight, 'method', 'unknown')
            
            if result == 'win':
                if method in ['kotko', 'ko', 'tko']:
                    score += 3
                elif method == 'sub':
                    score += 2.5
                elif method == 'd_unan':
                    score += 1.5
                elif method in ['d_maj', 'd_split']:
                    score += 1
                else:
                    score += 1
            elif result == 'loss':
                if method in ['kotko', 'ko', 'tko']:
                    score -= 2
                elif method == 'sub':
                    score -= 1.5
                elif method == 'd_unan':
                    score -= 0.75
                elif method in ['d_maj', 'd_split']:
                    score -= 0.5
        
        return float(score / len(fights_list))
    
    def calc_finish_percentage(fights_list):
        """% of wins that were finishes (KO or Sub)"""
        if not fights_list or len(fights_list) == 0:
            return None
        
        wins = [f for f in fights_list if f.get('result') == 'win']
        if len(wins) == 0:
            return 0.0
        
        finishes = sum(1 for f in wins if safe_get(f, 'method', '') in ['kotko', 'ko', 'tko', 'sub'])
        return float(finishes / len(wins))
    
    def calc_ko_power(fights_list):
        """% of wins by KO/TKO"""
        if not fights_list or len(fights_list) == 0:
            return None
        
        wins = [f for f in fights_list if f.get('result') == 'win']
        if len(wins) == 0:
            return 0.0
        
        kos = sum(1 for f in wins if safe_get(f, 'method', '') in ['kotko', 'ko', 'tko'])
        return float(kos / len(wins))
    
    def calc_sub_threat(fights_list):
        """% of wins by submission"""
        if not fights_list or len(fights_list) == 0:
            return None
        
        wins = [f for f in fights_list if f.get('result') == 'win']
        if len(wins) == 0:
            return 0.0
        
        subs = sum(1 for f in wins if safe_get(f, 'method', '') == 'sub')
        return float(subs / len(wins))
    
    def calc_decision_wins_pct(fights_list):
        """% of wins by decision"""
        if not fights_list or len(fights_list) == 0:
            return None
        
        wins = [f for f in fights_list if f.get('result') == 'win']
        if len(wins) == 0:
            return 0.0
        
        decisions = sum(1 for f in wins if safe_get(f, 'method', '') in ['d_unan', 'd_maj', 'd_split'])
        return float(decisions / len(wins))
    
    def calc_dominant_decision_rate(fights_list):
        """Of decision wins, what % were unanimous"""
        if not fights_list or len(fights_list) == 0:
            return None
        
        wins = [f for f in fights_list if f.get('result') == 'win']
        decision_wins = [f for f in wins if safe_get(f, 'method', '') in ['d_unan', 'd_maj', 'd_split']]
        
        if len(decision_wins) == 0:
            return None
        
        unanimous = sum(1 for f in decision_wins if safe_get(f, 'method', '') == 'd_unan')
        return float(unanimous / len(decision_wins))
    
    def calc_been_kod_rate(fights_list):
        """% of losses by KO/TKO (durability indicator)"""
        if not fights_list or len(fights_list) == 0:
            return None
        
        losses = [f for f in fights_list if f.get('result') == 'loss']
        if len(losses) == 0:
            return 0.0
        
        ko_losses = sum(1 for f in losses if safe_get(f, 'method', '') in ['kotko', 'ko', 'tko'])
        return float(ko_losses / len(losses))
    
    def calc_been_subbed_rate(fights_list):
        """% of losses by submission (grappling defense indicator)"""
        if not fights_list or len(fights_list) == 0:
            return None
        
        losses = [f for f in fights_list if f.get('result') == 'loss']
        if len(losses) == 0:
            return 0.0
        
        sub_losses = sum(1 for f in losses if safe_get(f, 'method', '') == 'sub')
        return float(sub_losses / len(losses))
    
    def calc_momentum_trend(fights_list, window=5):
        """
        Compare recent performance to earlier career.
        Positive = improving, negative = declining
        """
        if not fights_list or len(fights_list) < window * 2:
            return None
        
        recent = fights_list[-window:]
        earlier = fights_list[-window*2:-window]
        
        recent_wins = sum(1 for f in recent if f.get('result') == 'win') / len(recent)
        earlier_wins = sum(1 for f in earlier if f.get('result') == 'win') / len(earlier)
        
        return float(recent_wins - earlier_wins)
    
    def calc_recent_form(fights_list, n=3):
        """Calculate win rate in last n fights"""
        if not fights_list or len(fights_list) == 0:
            return None
        recent = fights_list[-n:] if len(fights_list) >= n else fights_list
        wins = sum(1 for f in recent if f.get('result') == 'win')
        return wins / len(recent)
    
    def calc_win_rate(fights_list):
        """Overall career win rate"""
        if not fights_list or len(fights_list) == 0:
            return None
        wins = sum(1 for f in fights_list if f.get('result') == 'win')
        return wins / len(fights_list)
    
    def calc_avg_stat(fights_list, stat_name):
        """Average of a specific stat across all fights"""
        if not fights_list or len(fights_list) == 0:
            return None
        values = [safe_get(f, stat_name, 0) for f in fights_list]
        return float(np.mean(values)) if values else None
    
    def calc_accuracy(fights_list, landed_stat, attempts_stat):
        """Calculate accuracy % for a stat"""
        if not fights_list or len(fights_list) == 0:
            return None
        total_landed = sum(safe_get(f, landed_stat, 0) for f in fights_list)
        total_attempts = sum(safe_get(f, attempts_stat, 0) for f in fights_list)
        return float(total_landed / total_attempts) if total_attempts > 0 else None
    
    def calc_avg_fight_pace(fights_list):
        """Average strikes per minute"""
        if not fights_list or len(fights_list) == 0:
            return None
        total_strikes = sum(safe_get(f, 'total_str_attempts', 0) for f in fights_list)
        return float(total_strikes / (len(fights_list) * 15)) if len(fights_list) > 0 else None
    
    def calc_defensive_stats(fights_list):
        """Average opponent striking accuracy (lower is better defense)"""
        if not fights_list or len(fights_list) == 0:
            return None
        total_landed = sum(safe_get(f, 'opp_sig_str_landed', 0) for f in fights_list)
        total_attempts = sum(safe_get(f, 'opp_sig_str_attempts', 0) for f in fights_list)
        return float(total_landed / total_attempts) if total_attempts > 0 else None
    
    # Process each row
    features_list = []
    
    for idx, row in enumerate(df.iter_rows(named=True)):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} fights...")
        
        prior_f1 = row.get('prior_f1', [])
        prior_f2 = row.get('prior_f2', [])
        current_date = row.get('fight_date')
        
        if prior_f1 is None:
            prior_f1 = []
        if prior_f2 is None:
            prior_f2 = []
        
        # Calculate recency weights
        f1_weights = calc_recency_weights(prior_f1, current_date, half_life_days=365)
        f2_weights = calc_recency_weights(prior_f2, current_date, half_life_days=365)
        
        features = {
            # Basic info
            'fight_id': row['fight_id'],
            'fighter1_id': row['fighter1_id'],
            'fighter2_id': row['fighter2_id'],
            'winner_id': row['winner_id'],
            'fight_date': row['fight_date'],
            
            # Copy over fighter attributes
            'f1_reach_in': row.get('f1_reach_in'),
            'f2_reach_in': row.get('f2_reach_in'),
            'f1_height_in': row.get('f1_height_in'),
            'f2_height_in': row.get('f2_height_in'),
            'f1_dob': row.get('f1_dob'),
            'f2_dob': row.get('f2_dob'),
            
            # Fighter 1 Features
            'f1_weighted_win_rate': calc_weighted_win_rate(prior_f1, f1_weights),
            'f1_weighted_sig_str_accuracy': calc_weighted_accuracy(prior_f1, 'sig_str_landed', 'sig_str_attempts', f1_weights),
            'f1_weighted_td_accuracy': calc_weighted_accuracy(prior_f1, 'td_landed', 'td_attempts', f1_weights),
            'f1_weighted_avg_sig_str_landed': calc_weighted_avg_stat(prior_f1, 'sig_str_landed', f1_weights),
            'f1_weighted_avg_ctrl_time': calc_weighted_avg_stat(prior_f1, 'ctrl_time_s', f1_weights),
            'f1_weighted_avg_td_landed': calc_weighted_avg_stat(prior_f1, 'td_landed', f1_weights),
            'f1_weighted_avg_head_str_landed': calc_weighted_avg_stat(prior_f1, 'head_landed', f1_weights),
            'f1_current_streak': calc_current_streak(prior_f1),
            'f1_momentum_trend': calc_momentum_trend(prior_f1),
            'f1_recent_form_3': calc_recent_form(prior_f1, 3),
            'f1_recent_form_5': calc_recent_form(prior_f1, 5),
            'f1_finish_quality_score': calc_finish_quality_score(prior_f1),
            'f1_finish_percentage': calc_finish_percentage(prior_f1),
            'f1_ko_power': calc_ko_power(prior_f1),
            'f1_sub_threat': calc_sub_threat(prior_f1),
            'f1_decision_wins_pct': calc_decision_wins_pct(prior_f1),
            'f1_dominant_decision_rate': calc_dominant_decision_rate(prior_f1),
            'f1_been_kod_rate': calc_been_kod_rate(prior_f1),
            'f1_been_subbed_rate': calc_been_subbed_rate(prior_f1),
            'f1_win_rate': calc_win_rate(prior_f1),
            'f1_sig_str_accuracy': calc_accuracy(prior_f1, 'sig_str_landed', 'sig_str_attempts'),
            'f1_avg_sig_str_landed': calc_avg_stat(prior_f1, 'sig_str_landed'),
            'f1_avg_sig_str_attempts': calc_avg_stat(prior_f1, 'sig_str_attempts'),
            'f1_avg_head_str_landed': calc_avg_stat(prior_f1, 'head_landed'),
            'f1_avg_body_str_landed': calc_avg_stat(prior_f1, 'body_landed'),
            'f1_avg_leg_str_landed': calc_avg_stat(prior_f1, 'leg_landed'),
            'f1_td_accuracy': calc_accuracy(prior_f1, 'td_landed', 'td_attempts'),
            'f1_avg_td_landed': calc_avg_stat(prior_f1, 'td_landed'),
            'f1_avg_td_attempts': calc_avg_stat(prior_f1, 'td_attempts'),
            'f1_avg_ctrl_time': calc_avg_stat(prior_f1, 'ctrl_time_s'),
            'f1_avg_sub_attempts': calc_avg_stat(prior_f1, 'sub_att'),
            'f1_avg_kds': calc_avg_stat(prior_f1, 'kd'),
            'f1_defensive_accuracy': calc_defensive_stats(prior_f1),
            'f1_avg_opp_sig_str_landed': calc_avg_stat(prior_f1, 'opp_sig_str_landed'),
            'f1_fight_pace': calc_avg_fight_pace(prior_f1),
            
            # Fighter 2 Features (same as F1)
            'f2_weighted_win_rate': calc_weighted_win_rate(prior_f2, f2_weights),
            'f2_weighted_sig_str_accuracy': calc_weighted_accuracy(prior_f2, 'sig_str_landed', 'sig_str_attempts', f2_weights),
            'f2_weighted_td_accuracy': calc_weighted_accuracy(prior_f2, 'td_landed', 'td_attempts', f2_weights),
            'f2_weighted_avg_sig_str_landed': calc_weighted_avg_stat(prior_f2, 'sig_str_landed', f2_weights),
            'f2_weighted_avg_ctrl_time': calc_weighted_avg_stat(prior_f2, 'ctrl_time_s', f2_weights),
            'f2_weighted_avg_td_landed': calc_weighted_avg_stat(prior_f2, 'td_landed', f2_weights),
            'f2_weighted_avg_head_str_landed': calc_weighted_avg_stat(prior_f2, 'head_landed', f2_weights),
            'f2_current_streak': calc_current_streak(prior_f2),
            'f2_momentum_trend': calc_momentum_trend(prior_f2),
            'f2_recent_form_3': calc_recent_form(prior_f2, 3),
            'f2_recent_form_5': calc_recent_form(prior_f2, 5),
            'f2_finish_quality_score': calc_finish_quality_score(prior_f2),
            'f2_finish_percentage': calc_finish_percentage(prior_f2),
            'f2_ko_power': calc_ko_power(prior_f2),
            'f2_sub_threat': calc_sub_threat(prior_f2),
            'f2_decision_wins_pct': calc_decision_wins_pct(prior_f2),
            'f2_dominant_decision_rate': calc_dominant_decision_rate(prior_f2),
            'f2_been_kod_rate': calc_been_kod_rate(prior_f2),
            'f2_been_subbed_rate': calc_been_subbed_rate(prior_f2),
            'f2_win_rate': calc_win_rate(prior_f2),
            'f2_sig_str_accuracy': calc_accuracy(prior_f2, 'sig_str_landed', 'sig_str_attempts'),
            'f2_avg_sig_str_landed': calc_avg_stat(prior_f2, 'sig_str_landed'),
            'f2_avg_sig_str_attempts': calc_avg_stat(prior_f2, 'sig_str_attempts'),
            'f2_avg_head_str_landed': calc_avg_stat(prior_f2, 'head_landed'),
            'f2_avg_body_str_landed': calc_avg_stat(prior_f2, 'body_landed'),
            'f2_avg_leg_str_landed': calc_avg_stat(prior_f2, 'leg_landed'),
            'f2_td_accuracy': calc_accuracy(prior_f2, 'td_landed', 'td_attempts'),
            'f2_avg_td_landed': calc_avg_stat(prior_f2, 'td_landed'),
            'f2_avg_td_attempts': calc_avg_stat(prior_f2, 'td_attempts'),
            'f2_avg_ctrl_time': calc_avg_stat(prior_f2, 'ctrl_time_s'),
            'f2_avg_sub_attempts': calc_avg_stat(prior_f2, 'sub_att'),
            'f2_avg_kds': calc_avg_stat(prior_f2, 'kd'),
            'f2_defensive_accuracy': calc_defensive_stats(prior_f2),
            'f2_avg_opp_sig_str_landed': calc_avg_stat(prior_f2, 'opp_sig_str_landed'),
            'f2_fight_pace': calc_avg_fight_pace(prior_f2),
        }
        
        features_list.append(features)
    
    return pl.DataFrame(features_list)


def calculate_feature_differentials(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the difference between fighter 1 and fighter 2 for each stat."""
    
    stats_to_diff = [
        'weighted_win_rate', 'weighted_sig_str_accuracy', 'weighted_td_accuracy',
        'weighted_avg_sig_str_landed', 'weighted_avg_ctrl_time', 'weighted_avg_td_landed',
        'weighted_avg_head_str_landed', 'current_streak', 'momentum_trend',
        'finish_quality_score', 'finish_percentage', 'ko_power', 'sub_threat',
        'decision_wins_pct', 'dominant_decision_rate', 'been_kod_rate', 'been_subbed_rate',
        'recent_form_3', 'recent_form_5', 'win_rate', 'sig_str_accuracy', 'avg_sig_str_landed',
        'avg_sig_str_attempts', 'avg_head_str_landed', 'avg_body_str_landed', 'avg_leg_str_landed',
        'td_accuracy', 'avg_td_landed', 'avg_td_attempts', 'avg_ctrl_time', 'avg_sub_attempts',
        'avg_kds', 'defensive_accuracy', 'avg_opp_sig_str_landed', 'fight_pace'
    ]
    
    for stat in stats_to_diff:
        f1_col = f'f1_{stat}'
        f2_col = f'f2_{stat}'
        diff_col = f'diff_{stat}'
        
        if f1_col in df.columns and f2_col in df.columns:
            df = df.with_columns([
                (pl.col(f1_col) - pl.col(f2_col)).alias(diff_col)
            ])
    
    return df