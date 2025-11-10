"""
Scrape schedule and game data from Barttorvik's super_sked endpoint
"""

import requests
import pandas as pd
import json
from datetime import datetime

# Super Sked column structure (55 columns total)
SUPER_SKED_COLUMNS = [
    # 0-7: Basic game info
    'muid',           # 0: Unique game ID (e.g., "DukeNorthCarolina11-15")
    'date',           # 1: Date in M/D/YY format (e.g., "11/10/25")
    'conmatch',       # 2: Conference match flag
    'matchup',        # 3: Matchup string (e.g., "Duke vs North Carolina")
    'prediction',     # 4: Predicted outcome
    'ttq',            # 5: Torvik Thrill Quotient (game excitement rating)
    'conf',           # 6: Conference
    'venue',          # 7: Venue (H=Home, A=Away, N=Neutral)

    # 8-18: Team 1 info
    'team1',          # 8: Team 1 name
    't1oe',           # 9: Team 1 offensive efficiency
    't1de',           # 10: Team 1 defensive efficiency
    't1py',           # 11: Team 1 pythag
    't1wp',           # 12: Team 1 win probability
    't1propt',        # 13: Team 1 projected
    'team2',          # 14: Team 2 name
    't2oe',           # 15: Team 2 offensive efficiency
    't2de',           # 16: Team 2 defensive efficiency
    't2py',           # 17: Team 2 pythag
    't2wp',           # 18: Team 2 win probability

    # 19-27: Team 2 continued + game info
    't2propt',        # 19: Team 2 projected
    'tpro',           # 20: Total projection
    't1qual',         # 21: Team 1 quality
    't2qual',         # 22: Team 2 quality
    'gp',             # 23: Games played
    'result',         # 24: Result (empty for upcoming games)
    'tempo',          # 25: Tempo
    'possessions',    # 26: Possessions
    't1pts',          # 27: Team 1 points (actual)

    # 28-36: Actual results
    't2pts',          # 28: Team 2 points (actual)
    'winner',         # 29: Winner name
    'loser',          # 30: Loser name
    't1adjt',         # 31: Team 1 adjusted tempo
    't2adjt',         # 32: Team 2 adjusted tempo
    't1adjo',         # 33: Team 1 adjusted offensive efficiency
    't1adjd',         # 34: Team 1 adjusted defensive efficiency
    't2adjo',         # 35: Team 2 adjusted offensive efficiency
    't2adjd',         # 36: Team 2 adjusted defensive efficiency

    # 37-45: Game quality metrics
    'gamevalue',      # 37: Game value
    'mismatch',       # 38: Mismatch indicator
    'blowout',        # 39: Blowout indicator
    't1elite',        # 40: Team 1 elite status
    't2elite',        # 41: Team 2 elite status
    'ord_date',       # 42: Ordinal date
    't1ppp',          # 43: Team 1 points per possession
    't2ppp',          # 44: Team 2 points per possession
    'gameppp',        # 45: Game points per possession

    # 46-54: Rankings and game stats
    't1rk',           # 46: Team 1 rank
    't2rk',           # 47: Team 2 rank
    't1gs',           # 48: Team 1 game score
    't2gs',           # 49: Team 2 game score
    'gamestats',      # 50: Game stats (complex field with box score data)
    'overtimes',      # 51: Number of overtimes
    't1fun',          # 52: Team 1 fun rating
    't2fun',          # 53: Team 2 fun rating
    'results',        # 54: Results field
]


def scrape_barttorvik_schedule(season: str):
    """
    Scrape the full schedule data from Barttorvik's super_sked endpoint for a given season.

    Args:
        season: Season year as string (e.g., '2025', '2026')

    Returns:
        DataFrame with schedule data (with proper column names) or None if failed

    Example:
        >>> df = scrape_barttorvik_schedule('2026')
        >>> print(len(df))
        >>> print(df[['team1', 'team2', 'date', 't1oe', 't2oe']])
    """
    url = f"https://barttorvik.com/{season}_super_sked.json"

    try:
        print(f"Fetching schedule for season {season}...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        data = response.json()

        if not data or len(data) == 0:
            print(f"No schedule data found for season {season}")
            return None

        print(f"Retrieved {len(data)} games")

        # Process the data - it's a list of lists
        # Each inner list contains game data with 55 columns
        df = pd.DataFrame(data)

        # Assign proper column names
        df.columns = SUPER_SKED_COLUMNS[:len(df.columns)]

        print(f"DataFrame shape: {df.shape}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching schedule from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def parse_game_date_yyyymmdd(date_str: str):
    """
    Parse game date string from Barttorvik format.

    Args:
        date_str: Date in YYYYMMDD format (e.g., "20261110")

    Returns:
        datetime object or None if parsing fails
    """
    try:
        return pd.to_datetime(date_str, format='%Y%m%d')
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None


def format_date_to_mmddyy(date_yyyymmdd: str) -> str:
    """
    Convert YYYYMMDD format to M/D/YY format for matching with other systems.

    Args:
        date_yyyymmdd: Date in YYYYMMDD format (e.g., "20261110")

    Returns:
        Date in M/D/YY format (e.g., "11/10/26")
    """
    try:
        dt = pd.to_datetime(date_yyyymmdd, format='%Y%m%d')
        return dt.strftime('%-m/%-d/%y').lstrip('0')
    except Exception:
        # Fallback for Windows that doesn't support %-m
        try:
            dt = pd.to_datetime(date_yyyymmdd, format='%Y%m%d')
            month = str(dt.month)
            day = str(dt.day)
            year = dt.strftime('%y')
            return f"{month}/{day}/{year}"
        except Exception as e:
            print(f"Error formatting date '{date_yyyymmdd}': {e}")
            return None


if __name__ == "__main__":
    # Test the scraper
    print("Testing Barttorvik schedule scraper...")

    # Try scraping 2026 season
    df = scrape_barttorvik_schedule('2026')

    if df is not None:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {df.columns[:15].tolist()}")
        print(f"\nFirst game:")
        print(f"  MUID: {df.iloc[0]['muid']}")
        print(f"  Date (YYYYMMDD): {df.iloc[0]['date']}")
        print(f"  Team 1: {df.iloc[0]['team1']}")
        print(f"  Team 2: {df.iloc[0]['team2']}")
        print(f"  T1 Offensive Eff: {df.iloc[0]['t1oe']}")
        print(f"  T2 Offensive Eff: {df.iloc[0]['t2oe']}")
        print(f"  Winner: {df.iloc[0]['winner']}")
