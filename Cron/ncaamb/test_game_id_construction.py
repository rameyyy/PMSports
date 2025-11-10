#!/usr/bin/env python3
"""
Test game_id construction from player stats
"""
import sys
import os

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes.playerstats import scrape_player_stats
from load_player_stats import construct_game_id

print("Testing game_id construction...")
print("="*80)

df = scrape_player_stats(year='2026')

if df is None or len(df) == 0:
    print("❌ No data to test")
else:
    print(f"Testing with {len(df)} records\n")

    # Test first 10 records
    successful = 0
    failed = 0
    failed_examples = []

    for idx, row in df.head(10).iterrows():
        numdate = row.get('numdate')
        datetext = row.get('datetext')
        team = row.get('tt')
        opponent = row.get('opponent')

        game_id = construct_game_id(str(numdate), team, opponent)

        if game_id:
            successful += 1
            print(f"✅ Row {idx}: {team} vs {opponent} ({datetext}) -> {game_id}")
        else:
            failed += 1
            failed_examples.append({
                'idx': idx,
                'numdate': numdate,
                'datetext': datetext,
                'team': team,
                'opponent': opponent
            })
            print(f"❌ Row {idx}: {team} vs {opponent} ({datetext}) -> None")

    print(f"\nResults: {successful} successful, {failed} failed")

    if failed_examples:
        print("\nFailed examples:")
        for ex in failed_examples:
            print(f"  Row {ex['idx']}: datetext={ex['datetext']}, team={ex['team']}, opponent={ex['opponent']}")
