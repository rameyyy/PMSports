#!/usr/bin/env python3
"""
Validate that odds in features CSV files are correctly mapped to team_1 and team_2
Compares database odds against features CSV to ensure team mapping is working correctly
"""

import mysql.connector
import polars as pl
import csv
import os
import sys
from dotenv import load_dotenv

# Add models path for odds conversion functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.overunder.ou_feature_build_utils import average_american_odds

load_dotenv()

def load_team_mappings():
    """Load team mappings from CSV"""
    team_mappings = {}
    mappings_path = os.path.join(os.path.dirname(__file__), 'bookmaker', 'team_mappings.csv')
    try:
        with open(mappings_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('from_odds_team_name') and row.get('my_team_name'):
                    team_mappings[row['from_odds_team_name']] = row['my_team_name']
        print(f"[+] Loaded team mappings for {len(team_mappings)} teams")
        return team_mappings
    except Exception as e:
        print(f"[!] Error loading team mappings: {e}")
        return {}


def get_sample_games(year):
    """Get sample of games from features CSV for validation"""
    features_file = f"features{year}.csv"
    if not os.path.exists(features_file):
        print(f"[!] File not found: {features_file}")
        return []

    df = pl.read_csv(features_file)

    # Filter games with moneyline data
    games_with_ml = df.filter(
        (pl.col('avg_ml_team_1').is_not_null()) &
        (pl.col('avg_ml_team_2').is_not_null())
    )

    # Take first 20 games for validation
    sample = games_with_ml.head(20)

    return sample.to_dicts()


def get_odds_for_game(cursor, game_id):
    """Get all odds for a specific game from database"""
    cursor.execute("""
        SELECT game_id, home_team, away_team, bookmaker,
               ml_home, ml_away,
               spread_home, spread_pts_home, spread_away, spread_pts_away,
               over_odds, under_odds, over_point, under_point
        FROM odds
        WHERE game_id = %s
        ORDER BY bookmaker
    """, (game_id,))

    return cursor.fetchall()


def normalize_bookmaker_name(bookmaker):
    """Normalize bookmaker name"""
    if not bookmaker:
        return ''

    normalizations = {
        'BetOnline.ag': 'betonline',
        'betonlineag': 'betonline',
        'MyBookie.ag': 'mybookie',
        'mybookieag': 'mybookie',
        'LowVig.ag': 'lowvig',
        'lowvig': 'lowvig',
        'BetMGM': 'betmgm',
        'Bovada': 'bovada',
        'DraftKings': 'draftkings',
        'FanDuel': 'fanduel',
        'Caesars': 'caesars',
        'Bookmaker': 'bookmaker'
    }

    return normalizations.get(bookmaker, bookmaker.lower().replace('.ag', '').replace(' ', ''))


def validate_game(game_features, db_odds, team_mappings):
    """
    Validate that odds for a game are correctly mapped to team_1/team_2
    Returns list of issues found
    """
    issues = []
    game_id = game_features['game_id']
    team_1 = game_features['team_1']
    team_2 = game_features['team_2']

    if not db_odds:
        issues.append(f"No odds found in database for game {game_id}")
        return issues

    # Group odds by bookmaker
    odds_by_book = {}
    for odds in db_odds:
        bookmaker = normalize_bookmaker_name(odds['bookmaker'])
        if bookmaker not in odds_by_book:
            odds_by_book[bookmaker] = []
        odds_by_book[bookmaker].append(odds)

    print(f"\n{'='*80}")
    print(f"Game: {game_id}")
    print(f"Team 1 (team_1): {team_1}")
    print(f"Team 2 (team_2): {team_2}")
    print(f"Database teams per book:")

    ml_team_1_list = []
    ml_team_2_list = []

    # Check each bookmaker
    for bookmaker in ['betmgm', 'betonline', 'bovada', 'draftkings', 'fanduel', 'lowvig', 'mybookie']:
        if bookmaker not in odds_by_book:
            continue

        odds = odds_by_book[bookmaker][0]  # Take first if multiple

        home_team_raw = odds['home_team']
        away_team_raw = odds['away_team']

        # Map team names
        home_team_mapped = team_mappings.get(home_team_raw, home_team_raw)
        away_team_mapped = team_mappings.get(away_team_raw, away_team_raw)

        # Determine which team is team_1 and team_2 based on mapping
        home_is_team_1 = (home_team_mapped == team_1)
        away_is_team_1 = (away_team_mapped == team_1)

        # Get moneylines (convert to float for comparison)
        ml_home = float(odds['ml_home']) if odds['ml_home'] is not None else None
        ml_away = float(odds['ml_away']) if odds['ml_away'] is not None else None

        print(f"\n  {bookmaker.upper()}:")
        print(f"    Home (DB): {home_team_raw} -> {home_team_mapped}")
        print(f"    Away (DB): {away_team_raw} -> {away_team_mapped}")

        if home_is_team_1:
            print(f"    Home is team_1 [OK]")
            print(f"      ml_team_1 (home): {ml_home}")
            print(f"      ml_team_2 (away): {ml_away}")
            if ml_home is not None:
                ml_team_1_list.append(ml_home)
            if ml_away is not None:
                ml_team_2_list.append(ml_away)
        elif away_is_team_1:
            print(f"    Away is team_1 [OK]")
            print(f"      ml_team_1 (away): {ml_away}")
            print(f"      ml_team_2 (home): {ml_home}")
            if ml_away is not None:
                ml_team_1_list.append(ml_away)
            if ml_home is not None:
                ml_team_2_list.append(ml_home)
        else:
            issue = f"{bookmaker}: Team mapping failed - {home_team_mapped}/{away_team_mapped} don't match {team_1}/{team_2}"
            issues.append(issue)
            print(f"    ERROR: Team mapping failed [FAIL]")

    # Compare with features CSV
    print(f"\nFeatures CSV values:")
    avg_ml_team_1 = game_features.get('avg_ml_team_1')
    avg_ml_team_2 = game_features.get('avg_ml_team_2')

    # Convert to float for comparison
    if avg_ml_team_1 is not None:
        avg_ml_team_1 = float(avg_ml_team_1)
    if avg_ml_team_2 is not None:
        avg_ml_team_2 = float(avg_ml_team_2)

    if avg_ml_team_1 is not None:
        print(f"  avg_ml_team_1: {avg_ml_team_1}")
    if avg_ml_team_2 is not None:
        print(f"  avg_ml_team_2: {avg_ml_team_2}")

    # Verify averages match (using decimal odds conversion method)
    if ml_team_1_list and avg_ml_team_1 is not None:
        expected_avg_1 = average_american_odds(ml_team_1_list)
        if expected_avg_1 is not None and abs(expected_avg_1 - avg_ml_team_1) > 0.1:
            issue = f"avg_ml_team_1: Expected {expected_avg_1:.2f}, got {avg_ml_team_1:.2f}"
            issues.append(issue)
            print(f"  WARNING: {issue}")
        else:
            print(f"  [OK] avg_ml_team_1 matches calculated average (decimal odds method)")

    if ml_team_2_list and avg_ml_team_2 is not None:
        expected_avg_2 = average_american_odds(ml_team_2_list)
        if expected_avg_2 is not None and abs(expected_avg_2 - avg_ml_team_2) > 0.1:
            issue = f"avg_ml_team_2: Expected {expected_avg_2:.2f}, got {avg_ml_team_2:.2f}"
            issues.append(issue)
            print(f"  WARNING: {issue}")
        else:
            print(f"  [OK] avg_ml_team_2 matches calculated average (decimal odds method)")

    return issues


def main():
    # Connect to database
    try:
        db = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('NCAAMB_DB')
        )
        cursor = db.cursor(dictionary=True)
        print("[+] Connected to database")
    except Exception as e:
        print(f"[!] Failed to connect to database: {e}")
        return

    # Load team mappings
    team_mappings = load_team_mappings()

    # Validate recent years
    years_to_check = ['2023', '2024', '2025']
    all_issues = {}

    for year in years_to_check:
        print(f"\n{'='*80}")
        print(f"VALIDATING FEATURES{year}.CSV")
        print(f"{'='*80}")

        sample_games = get_sample_games(year)

        if not sample_games:
            print(f"No games found for validation")
            continue

        year_issues = []

        for game_features in sample_games:
            game_id = game_features['game_id']
            db_odds = get_odds_for_game(cursor, game_id)

            game_issues = validate_game(game_features, db_odds, team_mappings)
            if game_issues:
                year_issues.extend(game_issues)

        all_issues[year] = year_issues

        if year_issues:
            print(f"\n[!] Found {len(year_issues)} issues in {year}:")
            for issue in year_issues:
                print(f"    - {issue}")
        else:
            print(f"\n[OK] All validation checks passed for {year}!")

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")

    total_issues = sum(len(issues) for issues in all_issues.values())
    if total_issues == 0:
        print("[OK] All odds mappings are correct!")
    else:
        print(f"[FAIL] Found {total_issues} issues total:")
        for year, issues in all_issues.items():
            if issues:
                print(f"\n  {year}: {len(issues)} issues")

    cursor.close()
    db.close()


if __name__ == "__main__":
    main()
