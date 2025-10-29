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

def get_missing_dates():
    """Get all dates between 11/03/2023 and 04/07/2024 that are NOT in the database"""
    season_start = datetime(2023, 11, 3)
    season_end = datetime(2024, 4, 7)
    today = datetime.now()

    # Only check up to today
    end_date = min(today, season_end)

    # Get dates that exist in database
    conn = sqlconn.create_connection()
    if not conn:
        print("❌ Could not connect to database")
        return []

    query = "SELECT DISTINCT DATE(date) as scrape_date FROM leaderboard"
    results = sqlconn.fetch(conn, query)
    conn.close()

    # Convert to set of date strings (YYYY-MM-DD format)
    existing_dates = set()
    for row in results:
        if row['scrape_date']:
            date_obj = row['scrape_date']
            if hasattr(date_obj, 'strftime'):
                existing_dates.add(date_obj.strftime('%Y-%m-%d'))
            else:
                existing_dates.add(str(date_obj))

    print(f"📊 Found {len(existing_dates)} dates already in database")

    # Generate ALL dates from 11/04/2024 to 04/07/2025 and remove those in database
    all_dates = []
    current = season_start

    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        all_dates.append(date_str)
        current += timedelta(days=1)

    # Remove existing dates from all dates
    missing_dates = [d for d in all_dates if d not in existing_dates]

    print(f"Total days in range (11/03/2023-04/07/2024): {len(all_dates)}")
    print(f"Missing dates to scrape: {len(missing_dates)}")

    return missing_dates

def run_season_scrapes():
    """Run the leaderboard scraper for missing dates in the season"""
    # Get only dates missing from database
    scrape_dates = get_missing_dates()
    total_scrapes = len(scrape_dates)

    if total_scrapes == 0:
        print("✅ Database is up to date - no missing dates found")
        return

    print(f"🏀 Starting NCAAMB season scrape")
    print(f"📊 Total missing dates to scrape: {total_scrapes}")
    if scrape_dates:
        print(f"📅 Date range: {scrape_dates[0]} to {scrape_dates[-1]}")
    print("-" * 60)

    successful = 0
    failed = 0

    for idx, date_str in enumerate(scrape_dates, 1):
        # Parse the date string to extract year, month, day
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = str(date_obj.year)
        end_date = date_obj.strftime('%m%d')
        percentage = (idx / total_scrapes) * 100

        print(f"\n[{idx}/{total_scrapes}] ({percentage:.1f}%) Scraping date: {date_str}")

        try:
            df = scrape_barttorvik_csv(year, end_date=end_date)
            if df is not None:
                print(f"  ✅ Success - {len(df)} teams processed")
                successful += 1
            else:
                print(f"  ❌ Failed - check logs above")
                failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1

        # Add a small delay between scrapes to be respectful to the server
        if idx < total_scrapes:
            time.sleep(2)

    print("\n" + "=" * 60)
    print(f"🏁 Season scrape complete!")
    print(f"  ✅ Successful: {successful}/{total_scrapes}")
    print(f"  ❌ Failed: {failed}/{total_scrapes}")
    print("=" * 60)

if __name__ == "__main__":
    run_season_scrapes()