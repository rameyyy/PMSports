"""
Analyze ROI performance by individual betting rules
"""

from scrapes.sqlconn import create_connection, fetch
from collections import defaultdict


def american_to_decimal(odds):
    """Convert American odds to decimal odds"""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def get_first_dog_pick_date(conn):
    """Find the date of the first pick with 'Dog' in betting_rule"""
    query = """
        SELECT MIN(date) as first_date
        FROM ncaamb.moneyline_picks
        WHERE LOWER(betting_rule) LIKE '%dog%'
    """
    result = fetch(conn, query)
    if result and result[0]['first_date']:
        return result[0]['first_date']
    return None


def get_picks_by_rule(conn, start_date):
    """
    Get all picks since start_date with outcomes, grouped by betting_rule.
    """
    query = """
        SELECT
            mp.game_id,
            mp.betting_rule,
            mp.date,
            m.team_1,
            m.team_2,
            m.gbm_prob_team_1,
            m.gbm_prob_team_2,
            m.best_book_odds_team_1,
            m.best_book_odds_team_2,
            m.winning_team
        FROM ncaamb.moneyline_picks mp
        JOIN ncaamb.moneyline m ON mp.game_id = m.game_id
        WHERE mp.date >= %s
          AND m.winning_team IS NOT NULL
          AND m.best_book_odds_team_1 IS NOT NULL
          AND m.best_book_odds_team_2 IS NOT NULL
        ORDER BY mp.betting_rule, mp.date, mp.game_id
    """
    return fetch(conn, query, (start_date,))


def analyze_by_rule(picks):
    """Group picks by betting_rule and calculate stats for each"""
    rules_data = defaultdict(list)

    # Group picks by rule
    for pick in picks:
        rules_data[pick['betting_rule']].append(pick)

    # Calculate stats for each rule
    results = []

    for rule, rule_picks in rules_data.items():
        total_picks = len(rule_picks)
        correct_picks = 0
        total_wagered = 0
        total_profit = 0

        for row in rule_picks:
            # Determine which team was picked
            is_underdog = row['betting_rule'].lower().startswith('dog')

            if is_underdog:
                if row['gbm_prob_team_1'] <= row['gbm_prob_team_2']:
                    picked_team = row['team_1']
                    picked_odds = int(row['best_book_odds_team_1'])
                else:
                    picked_team = row['team_2']
                    picked_odds = int(row['best_book_odds_team_2'])
            else:
                if row['gbm_prob_team_1'] > row['gbm_prob_team_2']:
                    picked_team = row['team_1']
                    picked_odds = int(row['best_book_odds_team_1'])
                else:
                    picked_team = row['team_2']
                    picked_odds = int(row['best_book_odds_team_2'])

            # Check if won
            won = picked_team == row['winning_team']
            if won:
                correct_picks += 1

            # Calculate profit/loss
            stake = 100
            total_wagered += stake

            if won:
                decimal_odds = american_to_decimal(picked_odds)
                profit = stake * (decimal_odds - 1)
                total_profit += profit
            else:
                total_profit -= stake

        # Calculate metrics
        accuracy = round(correct_picks / total_picks * 100, 2) if total_picks > 0 else 0.0
        roi = round(total_profit / total_wagered * 100, 2) if total_wagered > 0 else 0.0

        results.append({
            'rule': rule,
            'total_picks': total_picks,
            'wins': correct_picks,
            'losses': total_picks - correct_picks,
            'accuracy': accuracy,
            'wagered': total_wagered,
            'profit': total_profit,
            'roi': roi
        })

    # Sort by ROI descending
    results.sort(key=lambda x: x['roi'], reverse=True)

    return results


def main():
    print("=" * 80)
    print("BETTING RULE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    conn = create_connection()
    if not conn:
        print("ERROR: Failed to connect to database")
        return

    try:
        # Find first Dog pick date
        print("Finding analysis period...")
        first_date = get_first_dog_pick_date(conn)

        if not first_date:
            print("No picks with 'Dog' in betting_rule found")
            return

        print(f"Analysis period: {first_date} to present")
        print()

        # Get all picks
        print("Fetching picks data...")
        picks = get_picks_by_rule(conn, first_date)

        if not picks:
            print(f"No completed picks found since {first_date}")
            return

        print(f"Analyzing {len(picks)} completed picks...")
        print()

        # Analyze by rule
        results = analyze_by_rule(picks)

        # Display results
        print("=" * 80)
        print("PERFORMANCE BY BETTING RULE")
        print("=" * 80)
        print()

        for i, r in enumerate(results, 1):
            status = "✅" if r['roi'] > 0 else "❌"
            print(f"{status} #{i}. {r['rule']}")
            print(f"   Record: {r['wins']}-{r['losses']} ({r['accuracy']:.2f}%)")
            print(f"   Wagered: ${r['wagered']:,.2f}")
            print(f"   Profit: ${r['profit']:,.2f}")
            print(f"   ROI: {r['roi']:.2f}%")
            print()

        # Summary stats
        total_profit = sum(r['profit'] for r in results)
        total_wagered = sum(r['wagered'] for r in results)
        overall_roi = round(total_profit / total_wagered * 100, 2) if total_wagered > 0 else 0.0

        profitable_rules = sum(1 for r in results if r['roi'] > 0)

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Betting Rules: {len(results)}")
        print(f"Profitable Rules: {profitable_rules}")
        print(f"Unprofitable Rules: {len(results) - profitable_rules}")
        print(f"Overall ROI: {overall_roi:.2f}%")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
