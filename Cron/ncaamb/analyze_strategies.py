"""
Strategy Performance Analyzer

Analyzes all betting strategies on 2026 season data and returns their ROI.
Used by pick_of_day script to determine which strategies are currently profitable.
"""

from scrapes.sqlconn import create_connection, fetch
import pandas as pd

def american_to_decimal(odds):
    """Convert American odds to decimal"""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def american_to_implied_prob(odds):
    """Convert American odds to implied probability %"""
    if odds > 0:
        return (100 / (odds + 100)) * 100
    else:
        return (abs(odds) / (abs(odds) + 100)) * 100

def calculate_roi(bets_df):
    """Calculate ROI based on actual betting outcomes"""
    total_wagered = 0
    total_profit = 0

    for _, row in bets_df.iterrows():
        stake = 100
        total_wagered += stake

        if row['correct']:
            decimal_odds = american_to_decimal(row['predicted_odds'])
            profit = stake * (decimal_odds - 1)
            total_profit += profit
        else:
            total_profit -= stake

    if total_wagered == 0:
        return 0

    roi = (total_profit / total_wagered) * 100
    return roi

def get_predicted_info(row):
    """Get predicted team and its EV/odds"""
    if row['gbm_prob_team_1'] > row['gbm_prob_team_2']:
        return {
            'predicted_team': row['team_1'],
            'predicted_ev': float(row['best_ev_team_1']) if row['best_ev_team_1'] else None,
            'predicted_odds': int(row['best_book_odds_team_1']) if row['best_book_odds_team_1'] else None
        }
    else:
        return {
            'predicted_team': row['team_2'],
            'predicted_ev': float(row['best_ev_team_2']) if row['best_ev_team_2'] else None,
            'predicted_odds': int(row['best_book_odds_team_2']) if row['best_book_odds_team_2'] else None
        }

def get_underdog_info(row):
    """Get the underdog team (model's predicted loser) and its EV/odds/prob"""
    if row['gbm_prob_team_1'] > row['gbm_prob_team_2']:
        # team_2 is the underdog
        return {
            'underdog_team': row['team_2'],
            'underdog_prob': float(row['gbm_prob_team_2']) * 100,
            'underdog_ev': float(row['best_ev_team_2']) if row['best_ev_team_2'] else None,
            'underdog_odds': int(row['best_book_odds_team_2']) if row['best_book_odds_team_2'] else None
        }
    else:
        # team_1 is the underdog
        return {
            'underdog_team': row['team_1'],
            'underdog_prob': float(row['gbm_prob_team_1']) * 100,
            'underdog_ev': float(row['best_ev_team_1']) if row['best_ev_team_1'] else None,
            'underdog_odds': int(row['best_book_odds_team_1']) if row['best_book_odds_team_1'] else None
        }

# All strategies to test - script filters out negative ROI automatically
STRATEGY_DEFINITIONS = [
    # Favorite strategies: bet on the model's predicted winner
    {'name': '52-58% Implied + EV > 0%', 'min_implied': 52, 'max_implied': 58, 'min_ev': 0.01, 'side': 'favorite'},
    {'name': '60-65% Implied + EV > 0%', 'min_implied': 60, 'max_implied': 65, 'min_ev': 0.01, 'side': 'favorite'},
    {'name': '60-70% Implied + EV > 1%', 'min_implied': 60, 'max_implied': 70, 'min_ev': 1.0, 'side': 'favorite'},
    {'name': '60-70% Implied + EV > 0%', 'min_implied': 60, 'max_implied': 70, 'min_ev': 0.01, 'side': 'favorite'},
    {'name': '55-65% Implied + EV > 0%', 'min_implied': 55, 'max_implied': 65, 'min_ev': 0.01, 'side': 'favorite'},
    {'name': '55-65% Implied + EV > 1%', 'min_implied': 55, 'max_implied': 65, 'min_ev': 1.0, 'side': 'favorite'},
    {'name': '55-65% Implied + EV > 2%', 'min_implied': 55, 'max_implied': 65, 'min_ev': 2.0, 'side': 'favorite'},
    {'name': '54-60% Implied + EV > 0%', 'min_implied': 54, 'max_implied': 60, 'min_ev': 0.01, 'side': 'favorite'},
    {'name': '55-60% Implied + EV > 0%', 'min_implied': 55, 'max_implied': 60, 'min_ev': 0.01, 'side': 'favorite'},
    # Underdog strategies: bet on the model's predicted loser when odds offer +EV
    # Uses ensemble_prob of the underdog (0-50%) as the bucket range
    {'name': 'Dog 25-35% Prob + EV > 0%', 'min_prob': 25, 'max_prob': 35, 'min_ev': 0.01, 'side': 'underdog'},
    {'name': 'Dog 30-35% Prob + EV > 0%', 'min_prob': 30, 'max_prob': 35, 'min_ev': 0.01, 'side': 'underdog'},
    {'name': 'Dog 25-30% Prob + EV > 0%', 'min_prob': 25, 'max_prob': 30, 'min_ev': 0.01, 'side': 'underdog'},
    {'name': 'Dog 40-45% Prob + EV > 0%', 'min_prob': 40, 'max_prob': 45, 'min_ev': 0.01, 'side': 'underdog'},
    {'name': 'Dog 10-20% Prob + EV > 0%', 'min_prob': 10, 'max_prob': 20, 'min_ev': 0.01, 'side': 'underdog'},
]

def analyze_all_strategies():
    """
    Analyze all strategies on 2026 season data.
    Returns list of strategies sorted by ROI (only positive ROI).
    """

    # Fetch 2026 season data
    query = """
        SELECT
            team_1,
            team_2,
            gbm_prob_team_1,
            gbm_prob_team_2,
            best_ev_team_1,
            best_ev_team_2,
            best_book_odds_team_1,
            best_book_odds_team_2,
            winning_team
        FROM ncaamb.moneyline
        WHERE season = 2026
          AND winning_team IS NOT NULL
          AND gbm_prob_team_1 IS NOT NULL
          AND gbm_prob_team_2 IS NOT NULL
          AND best_book_odds_team_1 IS NOT NULL
          AND best_book_odds_team_2 IS NOT NULL
          AND best_book_odds_team_1 != best_book_odds_team_2
          AND best_ev_team_1 IS NOT NULL
          AND best_ev_team_2 IS NOT NULL
    """

    conn = create_connection()
    if not conn:
        print("❌ Failed to connect to database")
        return []

    data = fetch(conn, query)
    conn.close()

    if not data or len(data) == 0:
        print("❌ No data found for analysis")
        return []

    df = pd.DataFrame(data)

    # Process games for favorite strategies
    fav_processed = []
    dog_processed = []
    for _, row in df.iterrows():
        pred = get_predicted_info(row)
        if pred['predicted_ev'] is not None and pred['predicted_odds'] is not None:
            fav_processed.append({
                'predicted_team': pred['predicted_team'],
                'predicted_ev': pred['predicted_ev'],
                'predicted_odds': pred['predicted_odds'],
                'implied_prob': american_to_implied_prob(pred['predicted_odds']),
                'winning_team': row['winning_team'],
                'correct': pred['predicted_team'] == row['winning_team']
            })

        dog = get_underdog_info(row)
        if dog['underdog_ev'] is not None and dog['underdog_odds'] is not None:
            dog_processed.append({
                'predicted_team': dog['underdog_team'],
                'predicted_ev': dog['underdog_ev'],
                'predicted_odds': dog['underdog_odds'],
                'underdog_prob': dog['underdog_prob'],
                'implied_prob': american_to_implied_prob(dog['underdog_odds']),
                'winning_team': row['winning_team'],
                'correct': dog['underdog_team'] == row['winning_team']
            })

    df_fav = pd.DataFrame(fav_processed)
    df_dog = pd.DataFrame(dog_processed)

    # Analyze each strategy
    results = []

    for strategy in STRATEGY_DEFINITIONS:
        if strategy['side'] == 'favorite':
            filtered = df_fav[
                (df_fav['implied_prob'] >= strategy['min_implied']) &
                (df_fav['implied_prob'] < strategy['max_implied']) &
                (df_fav['predicted_ev'] > strategy['min_ev'])
            ]
        else:  # underdog
            filtered = df_dog[
                (df_dog['underdog_prob'] >= strategy['min_prob']) &
                (df_dog['underdog_prob'] < strategy['max_prob']) &
                (df_dog['predicted_ev'] > strategy['min_ev'])
            ]

        if len(filtered) < 10:  # Need at least 10 games
            continue

        num_games = len(filtered)
        num_wins = filtered['correct'].sum()
        win_rate = (num_wins / num_games) * 100
        roi = calculate_roi(filtered)

        # Only include if ROI is positive
        if roi <= 0:
            continue

        avg_ev = filtered['predicted_ev'].mean()

        result = {
            'name': strategy['name'],
            'side': strategy['side'],
            'min_ev': strategy['min_ev'],
            'roi': roi,
            'win_rate': win_rate,
            'num_games': num_games,
            'num_wins': num_wins,
            'avg_ev': avg_ev
        }
        if strategy['side'] == 'favorite':
            result['min_implied'] = strategy['min_implied']
            result['max_implied'] = strategy['max_implied']
        else:
            result['min_prob'] = strategy['min_prob']
            result['max_prob'] = strategy['max_prob']

        results.append(result)

    # Sort by ROI descending
    results.sort(key=lambda x: x['roi'], reverse=True)

    return results

if __name__ == "__main__":
    print("=" * 80)
    print("STRATEGY PERFORMANCE ANALYSIS - 2026 Season")
    print("=" * 80)
    print()

    strategies = analyze_all_strategies()

    if not strategies:
        print("⚠️  No profitable strategies found!")
    else:
        print(f"Found {len(strategies)} profitable strategies:\n")
        print(f"{'Rank':<6} {'Strategy':<35} {'ROI':<10} {'Win Rate':<12} {'Games':<10}")
        print("-" * 80)

        for i, s in enumerate(strategies, 1):
            print(f"{i:<6} {s['name']:<35} {s['roi']:>6.2f}%   {s['win_rate']:>6.2f}%     {s['num_wins']}/{s['num_games']}")

    print()
