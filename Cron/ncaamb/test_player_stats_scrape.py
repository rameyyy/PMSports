#!/usr/bin/env python3
"""
Quick test to see if player stats scraping is working
"""
import sys
import os

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes.playerstats import scrape_player_stats

print("Testing player stats scraper...")
print("="*80)

df = scrape_player_stats(year='2026')

if df is None:
    print("❌ scrape_player_stats returned None")
elif len(df) == 0:
    print("❌ scrape_player_stats returned empty DataFrame")
else:
    print(f"✅ scrape_player_stats returned {len(df)} records")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst row sample:")
    print(df.iloc[0])
    print(f"\nChecking key columns exist:")
    print(f"  'tt' column exists: {'tt' in df.columns}")
    print(f"  'opponent' column exists: {'opponent' in df.columns}")
    print(f"  'datetext' column exists: {'datetext' in df.columns}")
    print(f"  'pp' column exists: {'pp' in df.columns}")
