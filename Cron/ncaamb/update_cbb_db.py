from scrapes.leaderboard import scrape_barttorvik_csv
from scrapes import sqlconn
from datetime import datetime, timedelta
import time
import sys
import os

# Enable UTF-8 encoding on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

def get_missing_dates_for_season(year):
    """
    Get all dates for a season that are NOT in the database.

    Args:
        year: Season year (e.g., 2025 for 2024-2025 season)

    Returns:
        List of missing dates in YYYY-MM-DD format
    """
    # Seasons run from November 1 to April 5
    # 2024 season = Nov 1, 2023 - Apr 5, 2024
    season_start = datetime(year - 1, 11, 30)
    season_end = datetime(year, 4, 5)
    today = datetime.now()

    # Only check up to today
    end_date = min(today, season_end)

    # Get dates that exist in database
    conn = sqlconn.create_connection()
    if not conn:
        print("❌ Could not connect to database")
        return []

    # Get all dates from leaderboard and infer season from the date
    query = "SELECT DISTINCT DATE(date) as scrape_date FROM leaderboard"
    results = sqlconn.fetch(conn, query)
    conn.close()

    # Convert to set of date strings (YYYY-MM-DD format) that belong to this season
    existing_dates = set()
    for row in results:
        if row['scrape_date']:
            date_obj = row['scrape_date']
            if hasattr(date_obj, 'strftime'):
                date_str = date_obj.strftime('%Y-%m-%d')
            else:
                date_str = str(date_obj)

            # Parse the date to check if it belongs to this season
            try:
                date_parsed = datetime.strptime(date_str, '%Y-%m-%d')
                # Check if date is in season range (Nov of previous year to Apr of current year)
                if season_start <= date_parsed <= season_end:
                    existing_dates.add(date_str)
            except ValueError:
                pass

    print(f"  📊 Found {len(existing_dates)} dates already in database for season {year}")

    # Generate ALL dates for the season and remove those in database
    all_dates = []
    current = season_start

    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        all_dates.append(date_str)
        current += timedelta(days=1)

    # Remove existing dates from all dates
    missing_dates = [d for d in all_dates if d not in existing_dates]

    print(f"  📅 Total days in season ({season_start.strftime('%m/%d/%Y')}-{end_date.strftime('%m/%d/%Y')}): {len(all_dates)}")
    print(f"  ⏳ Missing dates to scrape: {len(missing_dates)}")

    return missing_dates

def run_season_scrapes():
    """Run the leaderboard scraper for multiple seasons (2020-2025)"""
    seasons_to_scrape = [2025]

    print("🏀 Starting NCAAMB multi-season scrape")
    print(f"📊 Seasons to scrape: {', '.join(map(str, seasons_to_scrape))}")
    print("=" * 60)

    total_successful = 0
    total_failed = 0

    for season_year in seasons_to_scrape:
        print(f"\n🔄 Processing season {season_year}...")
        print("-" * 60)

        # Get only dates missing from database for this season
        scrape_dates = get_missing_dates_for_season(season_year)
        total_scrapes = len(scrape_dates)

        if total_scrapes == 0:
            print(f"✅ Season {season_year} is up to date - no missing dates found")
            continue

        print(f"  Starting scrape for {total_scrapes} missing dates")
        print()

        successful = 0
        failed = 0

        for idx, date_str in enumerate(scrape_dates, 1):
            # Parse the date string to extract year, month, day
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

            # Determine season year (Nov-Dec dates belong to next year's season)
            # e.g., 2024-11-15 is part of the 2024-2025 season (year=2025)
            if date_obj.month >= 11:  # November or December
                year = str(date_obj.year + 1)
            else:  # January through October
                year = str(date_obj.year)

            end_date = date_obj.strftime('%m%d')
            percentage = (idx / total_scrapes) * 100

            print(f"  [{idx}/{total_scrapes}] ({percentage:.1f}%) Scraping {date_str}...", end=" ")

            try:
                df = scrape_barttorvik_csv(year, end_date=end_date)
                if df is not None:
                    print(f"✅ ({len(df)} teams)")
                    successful += 1
                    total_successful += 1
                else:
                    print(f"❌ Failed")
                    failed += 1
                    total_failed += 1
            except Exception as e:
                print(f"❌ Error: {e}")
                failed += 1
                total_failed += 1

            # Add a small delay between scrapes to be respectful to the server
            if idx < total_scrapes:
                time.sleep(2)

        print()
        print(f"  Season {season_year} complete: {successful} successful, {failed} failed")

    print("\n" + "=" * 60)
    print(f"🏁 Multi-season scrape complete!")
    print(f"  ✅ Total successful: {total_successful}")
    print(f"  ❌ Total failed: {total_failed}")
    print("=" * 60)

if __name__ == "__main__":
    run_season_scrapes()