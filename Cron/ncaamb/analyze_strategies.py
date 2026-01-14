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

# Define all strategies to test (50+ games historically)
STRATEGY_DEFINITIONS = [
    {'name': '52-58% Implied + EV > 0%', 'min_implied': 52, 'max_implied': 58, 'min_ev': 0.01},
    {'name': '60-65% Implied + EV > 0%', 'min_implied': 60, 'max_implied': 65, 'min_ev': 0.01},
    {'name': '60-70% Implied + EV > 1%', 'min_implied': 60, 'max_implied': 70, 'min_ev': 1.0},
    {'name': '60-70% Implied + EV > 0%', 'min_implied': 60, 'max_implied': 70, 'min_ev': 0.01},
    {'name': '55-65% Implied + EV > 0%', 'min_implied': 55, 'max_implied': 65, 'min_ev': 0.01},
    {'name': '55-65% Implied + EV > 1%', 'min_implied': 55, 'max_implied': 65, 'min_ev': 1.0},
    {'name': '55-65% Implied + EV > 2%', 'min_implied': 55, 'max_implied': 65, 'min_ev': 2.0},
    {'name': '54-60% Implied + EV > 0%', 'min_implied': 54, 'max_implied': 60, 'min_ev': 0.01},
    {'name': '55-60% Implied + EV > 0%', 'min_implied': 55, 'max_implied': 60, 'min_ev': 0.01},
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

    # Process games
    processed = []
    for _, row in df.iterrows():
        pred = get_predicted_info(row)
        if pred['predicted_ev'] is None or pred['predicted_odds'] is None:
            continue

        processed.append({
            'predicted_team': pred['predicted_team'],
            'predicted_ev': pred['predicted_ev'],
            'predicted_odds': pred['predicted_odds'],
            'implied_prob': american_to_implied_prob(pred['predicted_odds']),
            'winning_team': row['winning_team'],
            'correct': pred['predicted_team'] == row['winning_team']
        })

    df_processed = pd.DataFrame(processed)

    # Analyze each strategy
    results = []

    for strategy in STRATEGY_DEFINITIONS:
        # Filter games matching this strategy
        filtered = df_processed[
            (df_processed['implied_prob'] >= strategy['min_implied']) &
            (df_processed['implied_prob'] < strategy['max_implied']) &
            (df_processed['predicted_ev'] > strategy['min_ev'])
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

        results.append({
            'name': strategy['name'],
            'min_implied': strategy['min_implied'],
            'max_implied': strategy['max_implied'],
            'min_ev': strategy['min_ev'],
            'roi': roi,
            'win_rate': win_rate,
            'num_games': num_games,
            'num_wins': num_wins,
            'avg_ev': avg_ev
        })

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
