#!/usr/bin/env python3
"""
Quick runner script to build raw dataframe with all fights and round data.

Usage:
    cd /Users/clayramey/Documents/2025/Git-Repos/PMSports
    python Cron/ufc/build_raw_df_runner.py
"""

from Cron.ufc.models.build_raw_df_all_fights_with_rounds import run

if __name__ == "__main__":
    # Build raw dataframe with minimum 1 prior fight per fighter
    df = run(min_prior_fights=1)

    print("\n🎉 Done! Check the output file:")
    print("   fight_snapshots_all_with_rounds.parquet")
