#!/usr/bin/env python3
"""
Update odds records by:
1. Matching team names using team_mappings.csv
2. Mapping game_ids from ncaamb.games table
"""

import sys
import os

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from bookmaker.map_game_id import main as map_game_ids


def main():
    """Main orchestration function"""
    print("="*100)
    print("UPDATING ODDS TABLE WITH GAME IDS")
    print("="*100)
    print()

    # Run game_id mapping
    print("Step 1: Mapping game_ids from ncaamb.games table...")
    print("-"*100)
    success = map_game_ids()

    if not success:
        print("\nL Error during game_id mapping")
        return 1

    print("\n" + "="*100)
    print(" Odds update complete!")
    print("="*100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
