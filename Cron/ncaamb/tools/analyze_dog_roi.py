"""
Calculate ROI for moneyline_picks since the first pick with "Dog" in betting_rule
"""

from scrapes.sqlconn import create_connection, fetch


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


def get_picks_since_date(conn, start_date):
    """
    Get all picks since start_date with outcomes.
    Join with moneyline table to get odds and results.
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
        ORDER BY mp.date, mp.game_id
    """
    return fetch(conn, query, (start_date,))


def calculate_roi(picks):
    """Calculate ROI for all picks"""
    if not picks:
        return 0.0, 0, 0, 0, 0

    total_picks = len(picks)
    correct_picks = 0
    total_wagered = 0
    total_profit = 0

    for row in picks:
        # Determine which team was picked based on betting_rule
        is_underdog = row['betting_rule'].lower().startswith('dog')

        if is_underdog:
            # Pick the underdog (team with lower probability)
            if row['gbm_prob_team_1'] <= row['gbm_prob_team_2']:
                picked_team = row['team_1']
                picked_odds = int(row['best_book_odds_team_1'])
            else:
                picked_team = row['team_2']
                picked_odds = int(row['best_book_odds_team_2'])
        else:
            # Pick the favorite (team with higher probability)
            if row['gbm_prob_team_1'] > row['gbm_prob_team_2']:
                picked_team = row['team_1']
                picked_odds = int(row['best_book_odds_team_1'])
            else:
                picked_team = row['team_2']
                picked_odds = int(row['best_book_odds_team_2'])

        # Check if the pick won
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

    # Calculate ROI
    roi = round(total_profit / total_wagered * 100, 2) if total_wagered > 0 else 0.0
    accuracy = round(correct_picks / total_picks * 100, 2) if total_picks > 0 else 0.0

    return roi, accuracy, correct_picks, total_picks, total_profit


def main():
    print("=" * 80)
    print("ROI ANALYSIS: Picks Since First 'Dog' Strategy")
    print("=" * 80)
    print()

    conn = create_connection()
    if not conn:
        print("ERROR: Failed to connect to database")
        return

    try:
        # Step 1: Find first Dog pick date
        print("Step 1: Finding first 'Dog' pick...")
        print("-" * 80)
        first_date = get_first_dog_pick_date(conn)

        if not first_date:
            print("No picks with 'Dog' in betting_rule found")
            return

        print(f"First 'Dog' pick date: {first_date}")
        print()

        # Step 2: Get all picks since that date with outcomes
        print("Step 2: Fetching all picks since first Dog pick...")
        print("-" * 80)
        picks = get_picks_since_date(conn, first_date)

        if not picks:
            print(f"No completed picks found since {first_date}")
            return

        print(f"Found {len(picks)} completed picks")
        print()

        # Step 3: Calculate ROI
        print("Step 3: Calculating ROI...")
        print("-" * 80)
        roi, accuracy, correct, total, profit = calculate_roi(picks)

        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Period: {first_date} to present")
        print(f"Total Picks: {total}")
        print(f"Record: {correct}-{total - correct} ({accuracy}%)")
        print(f"Total Wagered: ${total * 100:,.2f} (${100} per pick)")
        print(f"Total Profit: ${profit:,.2f}")
        print(f"ROI: {roi}%")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
