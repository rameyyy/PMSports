#!/usr/bin/env python3
"""
Compare ncaamb.teams with odds table and update team_mappings.csv
"""

import sys
import os
import csv
from difflib import SequenceMatcher

# Add current directory to path for imports
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes.sqlconn import create_connection, fetch

MAPPINGS_CSV = os.path.join(ncaamb_dir, 'bookmaker', 'team_mappings.csv')


def get_ncaamb_teams():
    """Get all distinct team names from ncaamb.teams table"""
    conn = create_connection()
    if not conn:
        return set()

    try:
        query = "SELECT DISTINCT team_name FROM teams ORDER BY team_name"
        results = fetch(conn, query)
        teams = {row['team_name'] for row in results if row['team_name']}
        conn.close()
        return teams
    except Exception as e:
        print(f"Error fetching ncaamb.teams: {e}")
        if conn:
            conn.close()
        return set()


def get_odds_teams():
    """Get all distinct team names from odds table"""
    conn = create_connection()
    if not conn:
        return set()

    try:
        query = """
            SELECT DISTINCT home_team as team_name FROM odds
            UNION
            SELECT DISTINCT away_team as team_name FROM odds
        """
        results = fetch(conn, query)
        teams = {row['team_name'] for row in results if row['team_name']}
        conn.close()
        return teams
    except Exception as e:
        print(f"Error fetching odds teams: {e}")
        if conn:
            conn.close()
        return set()


def load_current_mappings():
    """Load existing mappings from CSV"""
    mappings = {}
    try:
        with open(MAPPINGS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('from_odds_team_name') and row.get('my_team_name'):
                    mappings[row['from_odds_team_name']] = row['my_team_name']
    except Exception as e:
        print(f"Error loading CSV: {e}")
    return mappings


def fuzzy_match(odds_team, ncaamb_teams, threshold=0.6):
    """Find best fuzzy match for an odds team name"""
    best_match = None
    best_score = 0

    for ncaamb_team in ncaamb_teams:
        ratio = SequenceMatcher(None, odds_team.lower(), ncaamb_team.lower()).ratio()
        if ratio > best_score and ratio >= threshold:
            best_score = ratio
            best_match = ncaamb_team

    return best_match, best_score


def find_missing_mappings(odds_teams, ncaamb_teams, current_mappings):
    """Find odds teams that don't have mappings"""
    missing = {}

    for odds_team in sorted(odds_teams):
        # Check if already mapped
        if odds_team in current_mappings:
            continue

        # Try fuzzy match
        best_match, score = fuzzy_match(odds_team, ncaamb_teams)

        missing[odds_team] = {
            'suggested': best_match,
            'score': score,
            'current_mapping': current_mappings.get(odds_team)
        }

    return missing


def update_csv_with_suggestions(missing_mappings, current_mappings, all_odds_teams):
    """Update CSV with new and suggested mappings"""
    all_mappings = current_mappings.copy()

    # Add new suggestions
    for odds_team, info in missing_mappings.items():
        if info['suggested']:
            all_mappings[odds_team] = info['suggested']
        else:
            all_mappings[odds_team] = ''

    # Write updated CSV
    try:
        with open(MAPPINGS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['from_odds_team_name', 'my_team_name'])
            writer.writeheader()
            for odds_name in sorted(all_odds_teams):
                writer.writerow({
                    'from_odds_team_name': odds_name,
                    'my_team_name': all_mappings.get(odds_name, '')
                })
        print(f"✓ Updated {MAPPINGS_CSV}")
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False


def main():
    """Main function"""
    print("="*100)
    print("FINDING MISSING TEAM MAPPINGS")
    print("="*100)
    print()

    print("Loading ncaamb.teams...")
    ncaamb_teams = get_ncaamb_teams()
    print(f"  Found {len(ncaamb_teams)} teams in ncaamb.teams")

    print("\nLoading odds table teams...")
    odds_teams = get_odds_teams()
    print(f"  Found {len(odds_teams)} teams in odds table")

    print("\nLoading current mappings...")
    current_mappings = load_current_mappings()
    print(f"  Found {len(current_mappings)} existing mappings")

    print("\nFinding missing mappings...")
    missing = find_missing_mappings(odds_teams, ncaamb_teams, current_mappings)
    print(f"  Found {len(missing)} odds teams without mappings")

    if missing:
        print("\n" + "-"*100)
        print("MISSING MAPPINGS (with fuzzy match suggestions):")
        print("-"*100)

        suggested_count = 0
        for odds_team in sorted(missing.keys()):
            info = missing[odds_team]
            suggested = info['suggested']
            score = info['score']

            if suggested:
                print(f"✓ {odds_team:<50} → {suggested:<30} (score: {score:.2f})")
                suggested_count += 1
            else:
                print(f"✗ {odds_team:<50} → NO MATCH FOUND")

        print(f"\n{suggested_count}/{len(missing)} missing mappings have suggestions")

        print("\nUpdating CSV with suggestions...")
        if update_csv_with_suggestions(missing, current_mappings, odds_teams):
            print("✓ CSV updated successfully!")
            print("\nReview the CSV and manually edit any incorrect mappings:")
            print(f"  {MAPPINGS_CSV}")
        else:
            print("✗ Error updating CSV")
            return False
    else:
        print("\n✓ All odds teams are already mapped!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
