#!/usr/bin/env python3
"""
Main entry point for NCAAMB data scraping and processing
"""

from scrapes.gamehistory import scrape_game_history

if __name__ == "__main__":
    year = '2025'
    team = 'Auburn'

    print(f"Fetching game history for {team} {year}\n")
    df = scrape_game_history(year=year, team=team)

    if df is not None:
        # Save to CSV in parent directory
        output_file = '../game_history_sample.csv'
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df)
    else:
        print("Failed to retrieve data")
