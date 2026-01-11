"""
Backfill script to scrape historical leaderboard data for a date range.
This script scrapes leaderboard data for each day between the specified start and end dates.
"""

from datetime import datetime, timedelta
from scrapes.leaderboard import scrape_barttorvik_csv
import time

def backfill_leaderboard_data(start_date_str, end_date_str, season_year):
    """
    Scrape leaderboard data for each day in the specified date range.

    Args:
        start_date_str: Start date in format "YYYY/MM/DD"
        end_date_str: End date in format "YYYY/MM/DD"
        season_year: The season year (e.g., "2026" for 2025-26 season)
    """
    # Parse dates
    start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
    end_date = datetime.strptime(end_date_str, "%Y/%m/%d")

    # Calculate total days
    total_days = (end_date - start_date).days + 1
    print(f"\n{'='*60}")
    print(f"Starting backfill for {total_days} days")
    print(f"Date range: {start_date_str} to {end_date_str}")
    print(f"Season: {season_year}")
    print(f"{'='*60}\n")

    current_date = start_date
    success_count = 0
    failed_dates = []

    while current_date <= end_date:
        # Format date as MMDD for the scraper
        end_date_param = current_date.strftime("%m%d")
        date_display = current_date.strftime("%Y/%m/%d")

        print(f"\n[{success_count + len(failed_dates) + 1}/{total_days}] Scraping data for {date_display}...")

        try:
            # Call the scraper with current date
            df = scrape_barttorvik_csv(
                year=season_year,
                output_dir='.',
                end_date=end_date_param
            )

            if df is not None and len(df) > 0:
                success_count += 1
                print(f"  [+] Successfully scraped {len(df)} teams for {date_display}")
            else:
                failed_dates.append(date_display)
                print(f"  [-] Failed to scrape data for {date_display}")

            # Wait between requests to avoid overwhelming the server
            if current_date < end_date:
                print("  [*] Waiting 3 seconds before next request...")
                time.sleep(3)

        except Exception as e:
            failed_dates.append(date_display)
            print(f"  [-] Error scraping {date_display}: {e}")

        # Move to next day
        current_date += timedelta(days=1)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Backfill Complete!")
    print(f"{'='*60}")
    print(f"Total days processed: {total_days}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_dates)}")

    if failed_dates:
        print(f"\nFailed dates:")
        for date in failed_dates:
            print(f"  - {date}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Backfill from December 9, 2025 to January 11, 2026
    # This is part of the 2025-26 season (season year 2026)
    backfill_leaderboard_data(
        start_date_str="2025/12/09",
        end_date_str="2026/01/11",
        season_year="2026"
    )
