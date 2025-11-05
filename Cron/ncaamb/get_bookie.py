"""
Fetch and push historical NCAA basketball odds for entire season.

Usage:
    python get_bookie.py <season_year>

Examples:
    python get_bookie.py 2025
    python get_bookie.py 2024

Season runs from Nov 22 to Mar 22, skipping holidays and non-game days.
Fetches odds at 9 AM EST daily.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

from bookmaker.get_data import fetch_odds_for_dates
from scrapes.sqlconn import create_connection, fetch, execute_query


# Holidays and breaks to skip (specific dates or date ranges)
SKIP_DATES = {
    2025: [
        # Christmas break (typically no games Dec 25-26)
        (datetime(2024, 12, 25), datetime(2024, 12, 26)),
        # New Year (typically no games Jan 1)
        (datetime(2025, 1, 1), datetime(2025, 1, 1)),
    ],
    2024: [
        (datetime(2023, 12, 25), datetime(2023, 12, 26)),
        (datetime(2024, 1, 1), datetime(2024, 1, 1)),
    ],
    2023: [
        (datetime(2022, 12, 25), datetime(2022, 12, 26)),
        (datetime(2023, 1, 1), datetime(2023, 1, 1)),
    ],
}


def get_season_dates(season_year: int) -> tuple:
    """
    Get start and end dates for NCAA basketball season.
    Season runs from Nov 22 (previous year) to Mar 22 (same year).

    Args:
        season_year: The year (e.g., 2025 for 2024-25 season)

    Returns:
        (start_date, end_date) as datetime objects
    """
    start_date = datetime(season_year - 1, 11, 22)
    end_date = datetime(season_year, 3, 22)
    return start_date, end_date


def should_skip_date(check_date: datetime, season_year: int) -> bool:
    """Check if a date should be skipped (holiday, break, etc.)"""
    skip_ranges = SKIP_DATES.get(season_year, [])

    for start, end in skip_ranges:
        if start <= check_date <= end:
            return True

    return False


def validate_season(season_year: int) -> bool:
    """Validate season year"""
    current_year = datetime.now().year
    if season_year < 2020 or season_year > current_year + 1:
        print(f"Error: Season year must be between 2020 and {current_year + 1}")
        return False
    return True


def push_odds_to_database(df: pd.DataFrame) -> bool:
    """Push odds DataFrame to ncaamb.odds table, updating nulls if record exists"""
    if df.empty:
        print("No data to push")
        return False

    try:
        from scrapes.sqlconn import create_connection, fetch

        # Select and rename columns to match database schema
        df_sql = pd.DataFrame()
        df_sql['game_id'] = None  # NULL for now
        df_sql['home_team'] = df['home_team']
        df_sql['away_team'] = df['away_team']
        # Convert datetime format from ISO to MySQL format (YYYY-MM-DD HH:MM:SS)
        df_sql['start_time'] = pd.to_datetime(df['start_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_sql['bookmaker'] = df['bookmaker_title']  # Use title instead of key
        df_sql['ml_home'] = pd.to_numeric(df['h2h_home'], errors='coerce')
        df_sql['ml_away'] = pd.to_numeric(df['h2h_away'], errors='coerce')
        df_sql['spread_home'] = pd.to_numeric(df['spread_home'], errors='coerce')
        df_sql['spread_pts_home'] = pd.to_numeric(df['spread_pts_home'], errors='coerce')
        df_sql['spread_away'] = pd.to_numeric(df['spread_away'], errors='coerce')
        df_sql['spread_pts_away'] = pd.to_numeric(df['spread_pts_away'], errors='coerce')
        df_sql['over_odds'] = pd.to_numeric(df['over_odds'], errors='coerce')
        df_sql['under_odds'] = pd.to_numeric(df['under_odds'], errors='coerce')
        df_sql['over_point'] = pd.to_numeric(df['over_point'], errors='coerce')
        df_sql['under_point'] = pd.to_numeric(df['under_point'], errors='coerce')

        # Separate into new records and existing records to update
        conn = create_connection()
        if not conn:
            print("Could not connect to database")
            return False

        new_rows = []
        updated_rows = []

        for idx, row in df_sql.iterrows():
            # Check if record exists
            check_query = """
                SELECT * FROM odds
                WHERE home_team = %s AND away_team = %s AND start_time = %s AND bookmaker = %s
            """
            existing = fetch(conn, check_query, (row['home_team'], row['away_team'], row['start_time'], row['bookmaker']))

            if existing:
                # Record exists - check for nulls to update
                existing_record = existing[0]
                update_fields = []
                update_values = []

                # Only update if new value is not null and existing value is null
                numeric_cols = ['ml_home', 'ml_away', 'spread_home', 'spread_pts_home',
                               'spread_away', 'spread_pts_away', 'over_odds', 'under_odds',
                               'over_point', 'under_point']

                for col in numeric_cols:
                    if pd.notna(row[col]) and existing_record.get(col) is None:
                        update_fields.append(f"{col} = %s")
                        update_values.append(row[col])

                if update_fields:
                    update_values.extend([row['home_team'], row['away_team'], row['start_time'], row['bookmaker']])
                    update_query = f"""
                        UPDATE odds SET {', '.join(update_fields)}
                        WHERE home_team = %s AND away_team = %s AND start_time = %s AND bookmaker = %s
                    """
                    updated_rows.append((update_query, update_values))
            else:
                # New record
                new_rows.append(row)

        # Insert new rows
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            success = execute_query(df=new_df, table_name='odds', if_exists='append')
            if not success:
                conn.close()
                return False
            print(f"  Inserted {len(new_rows)} new records")

        # Update existing rows with nulls
        if updated_rows:
            cursor = conn.cursor()
            for update_query, update_values in updated_rows:
                try:
                    cursor.execute(update_query, update_values)
                except Exception as e:
                    print(f"  Error updating record: {e}")
            conn.commit()
            print(f"  Updated {len(updated_rows)} existing records")
            cursor.close()

        conn.close()
        return True

    except Exception as e:
        print(f"Error pushing to database: {e}")
        return False


def main():
    """Main orchestration function"""

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python get_bookie.py <season_year>")
        print("Example: python get_bookie.py 2025")
        sys.exit(1)

    try:
        season_year = int(sys.argv[1])
    except ValueError:
        print("Error: Season year must be an integer")
        sys.exit(1)

    # Validate season
    if not validate_season(season_year):
        sys.exit(1)

    # Get season date range
    start_date, end_date = get_season_dates(season_year)
    print(f"Fetching NCAA Basketball odds for {season_year} season")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Time: 9:00 AM EST daily")
    print()

    # Calculate dates to fetch
    current = start_date
    dates_to_fetch = []

    while current <= end_date:
        if not should_skip_date(current, season_year):
            dates_to_fetch.append(current)
        current += timedelta(days=1)

    print(f"Total dates to fetch: {len(dates_to_fetch)}")
    print(f"(Skipping {(end_date - start_date).days + 1 - len(dates_to_fetch)} holidays/breaks)")
    print()

    # Fetch and push for each date
    total_inserted = 0
    total_updated = 0

    for i, fetch_date in enumerate(dates_to_fetch, 1):
        date_str = fetch_date.strftime("%Y-%m-%d")
        print(f"[{i}/{len(dates_to_fetch)}] {date_str} ... ", end="", flush=True)

        # Fetch odds for this date at 9 AM EST (2 PM UTC)
        odds_df = fetch_odds_for_dates(date_str, date_str, "14:00:00")

        if odds_df.empty:
            print("No games")
            continue

        print(f"{len(odds_df)} odds records ... ", end="", flush=True)

        # Push to database
        if push_odds_to_database(odds_df):
            print("Pushed")
        else:
            print("Failed")

    print()
    print("Season fetch complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
