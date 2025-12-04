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

ODDS_API_KEY1 = os.getenv("ODDS_API_KEY1")
ODDS_API_KEY2 = os.getenv("ODDS_API_KEY2")
ODDS_API_KEY = ODDS_API_KEY2  # Default to KEY2 (KEY1 being verified)
BASE_URL = "https://api.the-odds-api.com/v4"
USAGE_THRESHOLD = 460  # Switch to next key when usage >= 460


def get_active_api_key() -> str:
    """
    Determine which API key to use based on current usage.
    Prefers KEY2, switches to KEY1 if KEY2 reaches 460 requests.

    Returns:
        The active API key to use
    """
    global ODDS_API_KEY

    # Check KEY2 usage (primary key)
    key2_usage = check_api_usage(ODDS_API_KEY2)
    if key2_usage and key2_usage.get('requests_used') != 'N/A':
        key2_used = int(key2_usage.get('requests_used', 0))
        if key2_used < USAGE_THRESHOLD:
            ODDS_API_KEY = ODDS_API_KEY2
            return ODDS_API_KEY2
        else:
            print(f"[!] KEY2 usage ({key2_used}) >= {USAGE_THRESHOLD}. Switching to KEY1...")
            ODDS_API_KEY = ODDS_API_KEY1
            return ODDS_API_KEY1

    # Fallback to KEY1 if KEY2 check fails
    ODDS_API_KEY = ODDS_API_KEY1
    return ODDS_API_KEY1


def get_historical_odds(date_str: str = None) -> Optional[Dict]:
    """
    Fetch live NCAA basketball odds.
    Automatically switches API keys when one reaches usage threshold.

    Args:
        date_str: Optional parameter (kept for backwards compatibility, not used with live endpoint)

    Returns:
        JSON response from OddsAPI or None if request fails
    """
    # Get active API key (switches if KEY2 >= 460)
    active_key = get_active_api_key()
    if not active_key:
        print("Error: No valid ODDS_API keys found in .env file")
        return None

    # Use live endpoint instead of historical
    url = f"{BASE_URL}/sports/basketball_ncaab/odds"

    params = {
        "apiKey": active_key,
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
        print(f"Error fetching live odds: {e}")
        return None


def parse_odds_response(data: Dict, timestamp: str = None) -> pd.DataFrame:
    """
    Parse OddsAPI response into a structured DataFrame.
    Extracts odds for each bookmaker separately.
    Handles both historical (dict with 'data' key) and live (list) response formats.

    Args:
        data: Response from get_historical_odds()
        timestamp: Optional timestamp (for backwards compatibility)

    Returns:
        DataFrame with columns: game_id, timestamp, home_team, away_team, start_time,
                               bookmaker, h2h_home, h2h_away, spread_home, spread_pts_home,
                               spread_away, spread_pts_away, over_odds, under_odds, over_point, under_point
    """
    rows = []

    if not data:
        return pd.DataFrame()

    # Handle both response formats: historical (dict with 'data') and live (list)
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict) and 'data' in data:
        events = data.get('data', [])
    else:
        return pd.DataFrame()

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


def check_api_usage(api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Check current OddsAPI usage and quota by making a test request.
    Usage info is returned in response headers.

    Args:
        api_key: Optional API key to check. If None, uses current ODDS_API_KEY

    Returns:
        Dictionary with usage info or None if request fails
    """
    key_to_check = api_key or ODDS_API_KEY
    if not key_to_check:
        print("Error: ODDS_API key not found in .env file")
        return None

    # Make a minimal request to get usage headers
    url = f"{BASE_URL}/sports"
    params = {"apiKey": key_to_check}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Extract usage from headers
        usage_info = {
            "requests_used": response.headers.get('x-requests-used', 'N/A'),
            "requests_remaining": response.headers.get('x-requests-remaining', 'N/A'),
            "rate_limit": response.headers.get('x-ratelimit-limit', 'N/A'),
        }
        return usage_info
    except requests.exceptions.RequestException as e:
        print(f"Error checking API usage: {e}")
        return None


if __name__ == "__main__":
    # Check usage before for both keys
    print("Checking OddsAPI Usage BEFORE...")
    print("=" * 60)
    print("\nKEY1 Usage:")
    usage_key1_before = check_api_usage(ODDS_API_KEY1)
    key1_used_before = 0
    if usage_key1_before and usage_key1_before.get('requests_used') != 'N/A':
        key1_used_before = int(usage_key1_before.get('requests_used', 0))
        print(f"  Requests Used: {key1_used_before}")
        print(f"  Requests Remaining: {usage_key1_before.get('requests_remaining', 'N/A')}")
    else:
        print("  Failed to retrieve usage")

    print("\nKEY2 Usage:")
    usage_key2_before = check_api_usage(ODDS_API_KEY2)
    key2_used_before = 0
    if usage_key2_before and usage_key2_before.get('requests_used') != 'N/A':
        key2_used_before = int(usage_key2_before.get('requests_used', 0))
        print(f"  Requests Used: {key2_used_before}")
        print(f"  Requests Remaining: {usage_key2_before.get('requests_remaining', 'N/A')}")
    else:
        print("  Failed to retrieve usage")
    print("=" * 60)

    # Fetch odds for today
    print("\nFetching college basketball odds for today (2025-12-04)...")
    df = fetch_odds_for_dates("2025-12-04", "2025-12-04")
    print(f"Total records fetched: {len(df)}")
    if not df.empty:
        print(f"Games covered: {df['home_team'].nunique()}")
        print(f"Bookmakers: {df['bookmaker'].unique().tolist()}")

    # Check usage after for both keys
    print("\n" + "=" * 60)
    print("Checking OddsAPI Usage AFTER...")
    print("=" * 60)
    print("\nKEY1 Usage:")
    usage_key1_after = check_api_usage(ODDS_API_KEY1)
    key1_used_after = 0
    if usage_key1_after and usage_key1_after.get('requests_used') != 'N/A':
        key1_used_after = int(usage_key1_after.get('requests_used', 0))
        print(f"  Requests Used: {key1_used_after}")
        print(f"  Requests Remaining: {usage_key1_after.get('requests_remaining', 'N/A')}")
    else:
        print("  Failed to retrieve usage")

    print("\nKEY2 Usage:")
    usage_key2_after = check_api_usage(ODDS_API_KEY2)
    key2_used_after = 0
    if usage_key2_after and usage_key2_after.get('requests_used') != 'N/A':
        key2_used_after = int(usage_key2_after.get('requests_used', 0))
        print(f"  Requests Used: {key2_used_after}")
        print(f"  Requests Remaining: {usage_key2_after.get('requests_remaining', 'N/A')}")
    else:
        print("  Failed to retrieve usage")
    print("=" * 60)

    # Calculate cost
    key1_cost = key1_used_after - key1_used_before
    key2_cost = key2_used_after - key2_used_before
    print(f"\n[*] KEY1 Requests Cost: {key1_cost}")
    print(f"[*] KEY2 Requests Cost: {key2_cost}")
    print(f"[*] Total Requests Cost: {key1_cost + key2_cost}")
    print(f"[*] Current Active Key: {get_active_api_key()[:10]}... (KEY1)" if get_active_api_key() == ODDS_API_KEY1 else f"[*] Current Active Key: {get_active_api_key()[:10]}... (KEY2)")
    print(f"[*] NOT pushing to database (test only)")

