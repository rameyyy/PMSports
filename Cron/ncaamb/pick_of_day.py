"""
Moneyline Picks & Pick of the Day Script

1. Runs dynamic ROI analysis on all strategies using 2026 season data
2. Filters out strategies with negative ROI
3. Finds all today's games matching any profitable strategy
4. Inserts all matches into moneyline_picks table
5. Selects ONE as pick_of_day (highest ROI rule, then highest EV within that rule)
"""

from scrapes.sqlconn import create_connection, execute_query, fetch
from analyze_strategies import analyze_all_strategies, american_to_implied_prob
from datetime import datetime
import pytz

def get_todays_date_cst():
    """Get today's date in CST timezone"""
    cst = pytz.timezone('America/Chicago')
    current_time_cst = datetime.now(cst)
    return current_time_cst.strftime('%Y-%m-%d')

def get_predicted_info(row):
    """Determine which team is predicted and return relevant columns"""
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

def get_todays_games(today_date):
    """Fetch all today's games with predictions"""
    query = """
        SELECT
            m.game_id,
            m.team_1,
            m.team_2,
            m.gbm_prob_team_1,
            m.gbm_prob_team_2,
            m.best_ev_team_1,
            m.best_ev_team_2,
            m.best_book_odds_team_1,
            m.best_book_odds_team_2
        FROM ncaamb.moneyline m
        JOIN ncaamb.games g ON m.game_id = g.game_id
        WHERE g.date = %s
          AND m.gbm_prob_team_1 IS NOT NULL
          AND m.gbm_prob_team_2 IS NOT NULL
          AND m.best_book_odds_team_1 IS NOT NULL
          AND m.best_book_odds_team_2 IS NOT NULL
          AND m.best_book_odds_team_1 != m.best_book_odds_team_2
          AND m.best_ev_team_1 IS NOT NULL
          AND m.best_ev_team_2 IS NOT NULL
    """

    conn = create_connection()
    if not conn:
        return []

    games = fetch(conn, query, (today_date,))
    conn.close()
    return games if games else []

def find_all_picks(games, strategies):
    """
    Find all games matching any profitable strategy.
    Returns list of picks with their strategies.
    """
    all_picks = []

    for game in games:
        pred = get_predicted_info(game)

        if pred['predicted_ev'] is None or pred['predicted_odds'] is None:
            continue

        implied_prob = american_to_implied_prob(pred['predicted_odds'])

        # Check which strategies this game matches
        for strategy in strategies:
            if (implied_prob >= strategy['min_implied'] and
                implied_prob < strategy['max_implied'] and
                pred['predicted_ev'] > strategy['min_ev']):

                all_picks.append({
                    'game_id': game['game_id'],
                    'betting_rule': strategy['name'],
                    'predicted_ev': pred['predicted_ev'],
                    'implied_prob': implied_prob,
                    'strategy_roi': strategy['roi'],  # For sorting
                    'team_1': game['team_1'],
                    'team_2': game['team_2']
                })
                # Only add once per strategy (don't duplicate if multiple strategies match)
                break

    return all_picks

def select_pick_of_day(all_picks):
    """
    Select the pick of the day from all picks.
    Prioritize: highest ROI strategy, then within that, highest EV.
    """
    if not all_picks:
        return None

    # Sort by: 1) Strategy ROI (desc), 2) EV (desc)
    sorted_picks = sorted(all_picks, key=lambda x: (x['strategy_roi'], x['predicted_ev']), reverse=True)

    return sorted_picks[0]

def update_moneyline_picks_table(all_picks, pick_of_day_game_id, today_date):
    """
    Update moneyline_picks table:
    - Clear today's picks
    - Insert all picks
    - Mark pick_of_day=1 for the selected one
    """
    conn = create_connection()
    if not conn:
        print("ERROR: Failed to connect to database")
        return False

    cursor = conn.cursor()

    try:
        # Delete existing picks for today
        delete_query = "DELETE FROM ncaamb.moneyline_picks WHERE date = %s"
        cursor.execute(delete_query, (today_date,))

        # Insert all picks
        insert_query = """
            INSERT INTO ncaamb.moneyline_picks (game_id, betting_rule, pick_of_day, date, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """

        for pick in all_picks:
            is_potd = 1 if pick['game_id'] == pick_of_day_game_id else 0
            cursor.execute(insert_query, (pick['game_id'], pick['betting_rule'], is_potd, today_date))

        conn.commit()
        print(f"SUCCESS: Successfully inserted {len(all_picks)} picks into moneyline_picks")
        return True

    except Exception as e:
        print(f"ERROR: Error updating moneyline_picks table: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    """Main execution function"""
    print("=" * 80)
    print("MONEYLINE PICKS & PICK OF THE DAY SELECTION")
    print("=" * 80)
    print()

    today = get_todays_date_cst()
    print(f"Today's date (CST): {today}\n")

    # Step 1: Run dynamic strategy analysis
    print("Step 1: Analyzing strategy performance on 2026 season...")
    print("-" * 80)
    profitable_strategies = analyze_all_strategies()

    if not profitable_strategies:
        print("WARNING:  No profitable strategies found. Exiting.")
        return

    print(f"SUCCESS: Found {len(profitable_strategies)} profitable strategies")
    for i, s in enumerate(profitable_strategies[:5], 1):
        print(f"   {i}. {s['name']}: {s['roi']:.2f}% ROI ({s['num_wins']}/{s['num_games']})")
    print()

    # Step 2: Get today's games
    print("Step 2: Fetching today's games...")
    print("-" * 80)
    todays_games = get_todays_games(today)

    if not todays_games:
        print(f"WARNING:  No games with predictions found for {today}")
        return

    print(f"SUCCESS: Found {len(todays_games)} games with predictions\n")

    # Step 3: Find all picks matching profitable strategies
    print("Step 3: Matching games to profitable strategies...")
    print("-" * 80)
    all_picks = find_all_picks(todays_games, profitable_strategies)

    if not all_picks:
        print("WARNING:  No games match any profitable strategy")
        return

    print(f"SUCCESS: Found {len(all_picks)} picks across all strategies\n")

    # Step 4: Select pick of the day
    print("Step 4: Selecting Pick of the Day...")
    print("-" * 80)
    potd = select_pick_of_day(all_picks)

    if potd:
        print(f"SUCCESS: PICK OF THE DAY:")
        print(f"   Game: {potd['team_1']} vs {potd['team_2']}")
        print(f"   Strategy: {potd['betting_rule']}")
        print(f"   EV: {potd['predicted_ev']:.2f}%")
        print(f"   Implied Prob: {potd['implied_prob']:.1f}%")
        print(f"   Strategy ROI: {potd['strategy_roi']:.2f}%")
        print()

    # Step 5: Update database
    print("Step 5: Updating database...")
    print("-" * 80)
    success = update_moneyline_picks_table(all_picks, potd['game_id'] if potd else None, today)

    if success:
        print(f"\n{'=' * 80}")
        print(f"SUCCESS!")
        print(f"  Total Picks: {len(all_picks)}")
        print(f"  Pick of the Day: {potd['game_id'] if potd else 'None'}")
        print(f"{'=' * 80}")
    else:
        print("\nERROR: Failed to update database")
        exit(1)

    print()

if __name__ == "__main__":
    main()
