#!/usr/bin/env python3
"""Check if team_1 and team_2 are in alphabetical order in training data"""

import pandas as pd
from pathlib import Path

ncaamb_dir = Path(__file__).parent

# Check features files for each year
files_to_check = [
    'features2021.csv',
    'features2022.csv',
    'features2023.csv',
    'features2024.csv',
    'features2025.csv',
]

print("=" * 80)
print("CHECKING TEAM ORDERING IN TRAINING DATA")
print("=" * 80 + "\n")

for filename in files_to_check:
    filepath = ncaamb_dir / filename

    if not filepath.exists():
        print(f"[-] {filename}: FILE NOT FOUND")
        continue

    print(f"Checking {filename}...")

    try:
        # Read CSV
        df = pd.read_csv(filepath)

        # Check if team_1 and team_2 columns exist
        if 'team_1' not in df.columns or 'team_2' not in df.columns:
            print(f"    [-] Missing team_1 or team_2 columns")
            continue

        total_games = len(df)

        # Check if team_1 is alphabetically <= team_2 for each game
        alphabetical_count = 0
        non_alphabetical_games = []

        for idx, row in df.iterrows():
            team_1 = str(row['team_1']).strip()
            team_2 = str(row['team_2']).strip()

            if team_1 <= team_2:  # Alphabetically ordered
                alphabetical_count += 1
            else:
                non_alphabetical_games.append((team_1, team_2))

        alphabetical_pct = (alphabetical_count / total_games * 100) if total_games > 0 else 0

        print(f"    Total games: {total_games}")
        print(f"    Alphabetically ordered (team_1 <= team_2): {alphabetical_count} ({alphabetical_pct:.1f}%)")
        print(f"    NOT alphabetically ordered: {len(non_alphabetical_games)} ({100-alphabetical_pct:.1f}%)")

        if non_alphabetical_games and len(non_alphabetical_games) <= 10:
            print(f"\n    Examples of non-alphabetical ordering:")
            for team_1, team_2 in non_alphabetical_games[:10]:
                print(f"      {team_1} vs {team_2} (should be: {team_2} vs {team_1})")
        elif non_alphabetical_games:
            print(f"\n    First 10 examples of non-alphabetical ordering:")
            for team_1, team_2 in non_alphabetical_games[:10]:
                print(f"      {team_1} vs {team_2} (should be: {team_2} vs {team_1})")

        print()

    except Exception as e:
        print(f"    [-] Error reading file: {e}")
        print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If team_1 is ALWAYS alphabetically first (100%):
  → Models were trained with alphabetical ordering
  → Use alphabetical ordering for predictions

If team_1 is NEVER alphabetically first (0%):
  → Models were trained with REVERSE alphabetical ordering (team_2 < team_1)
  → Need to swap team_1/team_2 assignment

If team_1 is SOMETIMES alphabetically first:
  → Models may have been trained with schedule order or home/away logic
  → Need to investigate further
""")
