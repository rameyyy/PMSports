from scrapes.leaderboard import scrape_barttorvik_csv
from datetime import datetime, timedelta
import time

def get_all_scrape_dates():
    """Get all Mon/Wed/Thu/Fri/Sat dates in the 2024-2025 season"""
    season_start = datetime(2024, 11, 1)
    season_end = datetime(2025, 4, 8)
    today = datetime.now()

    # Only scrape up to today if we're in season, otherwise full season
    end_date = min(today, season_end)

    scrape_dates = []
    current = season_start

    while current <= end_date:
        if current.weekday() in [0, 2, 3, 4, 5]:  # Mon/Wed/Thu/Fri/Sat
            scrape_dates.append(current)
        current += timedelta(days=1)

    return scrape_dates

def run_season_scrapes():
    """Run the leaderboard scraper for all valid dates in the season"""
    scrape_dates = get_all_scrape_dates()
    total_scrapes = len(scrape_dates)

    print(f"🏀 Starting 2025 NCAAMB season scrape")
    print(f"📊 Total scrapes to complete: {total_scrapes}")
    print(f"📅 Date range: {scrape_dates[0].strftime('%Y-%m-%d')} to {scrape_dates[-1].strftime('%Y-%m-%d')}")
    print("-" * 60)

    successful = 0
    failed = 0

    for idx, scrape_date in enumerate(scrape_dates, 1):
        end_date = scrape_date.strftime('%m%d')
        percentage = (idx / total_scrapes) * 100

        print(f"\n[{idx}/{total_scrapes}] ({percentage:.1f}%) Scraping date: {scrape_date.strftime('%Y-%m-%d')}")

        try:
            df = scrape_barttorvik_csv('2025', end_date=end_date)
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