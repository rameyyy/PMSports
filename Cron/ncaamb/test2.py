#!/usr/bin/env python3
"""
Test scraper for Alabama St. to see data format
"""

from scrapes.gamehistory import scrape_game_history
import pandas as pd

if __name__ == "__main__":
    year = '2025'
    team = 'Alabama St.'

    print(f"Fetching game history for {team} {year}\n")
    df = scrape_game_history(year=year, team=team)

    if df is not None:
        print(f"Retrieved {len(df)} games")

        # Find Auburn game
        auburn_games = df[df['opponent'] == 'Auburn']
        if len(auburn_games) > 0:
            print(f"\nFound {len(auburn_games)} Auburn games")
            for idx, game in auburn_games.iterrows():
                print(f"\n=== Auburn Game {idx} ===")
                print(f"Date: {game['date']}")
                print(f"Team: {game['team']} Score: {game['team_score']}")
                print(f"Opponent: {game['opponent']} Score: {game['opp_score']}")
                print(f"\nTeam Stats:")
                print(f"  FGM/FGA: {game['team_fgm']}/{game['team_fga']}")
                print(f"  3PM/3PA: {game['team_3pm']}/{game['team_3pa']}")
                print(f"  FTM/FTA: {game['team_ftm']}/{game['team_fta']}")
                print(f"\nOpponent Stats:")
                print(f"  FGM/FGA: {game['opp_fgm']}/{game['opp_fga']}")
                print(f"  3PM/3PA: {game['opp_3pm']}/{game['opp_3pa']}")
                print(f"  FTM/FTA: {game['opp_ftm']}/{game['opp_fta']}")
        else:
            print("No Auburn games found")
            print("\nFirst game columns and values:")
            first_game = df.iloc[0]
            for col in df.columns:
                print(f"{col:<30} {first_game[col]}")
    else:
        print("Failed to retrieve data")
