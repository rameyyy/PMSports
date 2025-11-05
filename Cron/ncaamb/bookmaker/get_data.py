"""
Fetch historical NCAA basketball odds from OddsAPI.
Gets spreads, moneyline, and over/under data for a given timestamp.
"""

import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional

# Load environment variables
ncaamb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(ncaamb_dir, '.env')
load_dotenv(env_path)

ODDS_API_KEY = os.getenv("ODDS_API")
BASE_URL = "https://api.the-odds-api.com/v4"


def get_historical_odds(date_str: str) -> Optional[Dict]:
    """
    Fetch historical NCAA basketball odds for a given timestamp.

    Args:
        date_str: ISO8601 timestamp string (e.g., '2021-10-18T21:00:00Z')

    Returns:
        JSON response from OddsAPI or None if request fails
    """
    if not ODDS_API_KEY:
        print("Error: ODDS_API key not found in .env file")
        return None

    url = f"{BASE_URL}/historical/sports/basketball_ncaab/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "date": date_str,
        "regions": "us",
        "markets": "spreads,h2h,totals",
        "oddsFormat": "american",
        "bookmakers": "bookmaker,draftkings,fanduel,betmgm,caesars,betonlineag,bovada,mybookieag,lowvig"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds for {date_str}: {e}")
        return None


def parse_odds_response(data: Dict, timestamp: str) -> pd.DataFrame:
    """
    Parse OddsAPI response into a structured DataFrame.
    Extracts odds for each bookmaker separately.

    Args:
        data: Response from get_historical_odds()
        timestamp: The requested timestamp

    Returns:
        DataFrame with columns: game_id, timestamp, home_team, away_team, start_time,
                               bookmaker, h2h_home, h2h_away, spread_home, spread_pts_home,
                               spread_away, spread_pts_away, over_odds, under_odds, over_point, under_point
    """
    rows = []

    if not data or 'data' not in data:
        return pd.DataFrame()

    events = data.get('data', [])

    for event in events:
        try:
            game_id = event.get('id')
            start_time = event.get('commence_time')
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')

            if not home_team or not away_team:
                continue

            # Extract odds from each bookmaker
            for bookmaker in event.get('bookmakers', []):
                bookmaker_key = bookmaker.get('key', '')
                bookmaker_title = bookmaker.get('title', '')

                # Initialize odds for this bookmaker
                h2h_home = None
                h2h_away = None
                spread_home = None
                spread_pts_home = None
                spread_away = None
                spread_pts_away = None
                over_odds = None
                under_odds = None
                over_point = None
                under_point = None

                # Extract odds from markets
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')
                    outcomes = market.get('outcomes', [])

                    if market_key == 'h2h':  # Moneyline
                        for outcome in outcomes:
                            name = outcome.get('name', '')
                            price = outcome.get('price')
                            if name == home_team:
                                h2h_home = price
                            elif name == away_team:
                                h2h_away = price

                    elif market_key == 'spreads':  # Point spreads
                        for outcome in outcomes:
                            name = outcome.get('name', '')
                            price = outcome.get('price')
                            point = outcome.get('point')
                            if name == home_team:
                                spread_home = price
                                spread_pts_home = point
                            elif name == away_team:
                                spread_away = price
                                spread_pts_away = point

                    elif market_key == 'totals':  # Over/Under
                        for outcome in outcomes:
                            name = outcome.get('name', '')
                            price = outcome.get('price')
                            point = outcome.get('point')
                            if name == 'Over':
                                over_odds = price
                                over_point = point
                            elif name == 'Under':
                                under_odds = price
                                under_point = point

                # Only add row if this bookmaker has at least some odds
                if any([h2h_home, h2h_away, spread_home, spread_away, over_odds, under_odds]):
                    row = {
                        'game_id': game_id,
                        'timestamp': timestamp,
                        'home_team': home_team,
                        'away_team': away_team,
                        'start_time': start_time,
                        'bookmaker': bookmaker_key,
                        'bookmaker_title': bookmaker_title,
                        'h2h_home': h2h_home,
                        'h2h_away': h2h_away,
                        'spread_home': spread_home,
                        'spread_pts_home': spread_pts_home,
                        'spread_away': spread_away,
                        'spread_pts_away': spread_pts_away,
                        'over_odds': over_odds,
                        'under_odds': under_odds,
                        'over_point': over_point,
                        'under_point': under_point
                    }
                    rows.append(row)

        except Exception as e:
            print(f"Error parsing event {event.get('id')}: {e}")
            continue

    return pd.DataFrame(rows)


def fetch_odds_for_dates(start_date: str, end_date: str, time_of_day: str = "21:00:00") -> pd.DataFrame:
    """
    Fetch odds for all dates in a range.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        time_of_day: Time to fetch odds at (HH:MM:SS), default 9 PM

    Returns:
        Combined DataFrame of all odds
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_odds = []
    current = start

    while current <= end:
        iso_timestamp = f"{current.strftime('%Y-%m-%d')}T{time_of_day}Z"

        data = get_historical_odds(iso_timestamp)
        if data:
            df = parse_odds_response(data, iso_timestamp)
            if not df.empty:
                all_odds.append(df)

        current += timedelta(days=1)

    if all_odds:
        return pd.concat(all_odds, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Test
    df = fetch_odds_for_dates("2020-11-16", "2020-11-20")
    print(f"\nTotal records: {len(df)}")
    print(df.head())
