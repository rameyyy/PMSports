import undetected_chromedriver as uc
import pandas as pd
import time
import os
import sys

# Handle imports for both package and direct execution
try:
    from . import sqlconn
except ImportError:
    # Running directly, adjust path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scrapes import sqlconn

def scrape_barttorvik_csv(year='2024', output_dir='.', end_date=None):
    # Column headers - raw CSV columns (will be reordered and renamed)
    raw_columns = [
        'team',         # 0: Team name
        'adjoe',        # 1: Adjusted Offensive Efficiency
        'adjde',        # 2: Adjusted Defensive Efficiency
        'barthag',      # 3: Power Rating
        'rec',          # 4: Record (will split into wins/losses)
        'wins',         # 5: Wins
        'col6',         # 6: Total games (drop)
        'efg_off_prcnt',# 7
        'efg_def_prcnt',# 8
        'ftr',          # 9
        'ftrd',         # 10
        'tor',          # 11
        'tord',         # 12
        'orb',          # 13
        'drb',          # 14
        'col15',        # 15: Drop
        '2p_prcnt_off', # 16
        '2p_prcnt_def', # 17
        '3p_prcnt_off', # 18
        '3p_prcnt_def', # 19
        'col20',        # 20: Drop (the 0s)
        'col21',        # 21: Drop (the 0s)
        'col22',        # 22: Drop
        'col23',        # 23: Drop
        '3pr',          # 24: 3-Point Rate (40.2 in example)
        '3prd',         # 25: 3-Point Rate Defense (32.9 in example)
        'adj_t',        # 26: Adjusted Tempo (64.6459)
        'col27',        # 27: Drop (dupe of adj_t)
        'col28',        # 28: Drop (dupe)
        'col29',        # 29: Drop (dupe)
        'col30',        # 30: Drop (dupe)
        'col31',        # 31: Drop (dupe)
        'col32',        # 32: Drop (dupe)
        'col33',        # 33: Drop (dupe)
        'wab',          # 34: Wins Above Bubble
        'col35',        # 35: Drop
        'col36'         # 36: Drop
    ]
    
    # Setup download directory - ensure it exists
    download_dir = os.path.abspath(output_dir)
    os.makedirs(download_dir, exist_ok=True)

    # Build date parameters
    begin_date = f"{int(year)-1}1101"  # November 1st of previous year
    if end_date:
        # end_date should be in format "MMDD" (e.g., "0109" for January 9th)
        # Season spans two calendar years: Nov-Dec uses previous year, Jan+ uses current year
        month = int(end_date[:2])
        if month >= 11:  # November or December
            end_date_full = f"{int(year)-1}{end_date}"
        else:  # January through October
            end_date_full = f"{year}{end_date}"
    else:
        # Default to current year's date (for simplicity, use end of season)
        end_date_full = f"{year}0630"  # June 30th as default

    def create_chrome_options(download_dir):
        """Create a fresh ChromeOptions object with download preferences"""
        options = uc.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-dev-shm-usage')

        # Set download preferences
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False
        }
        options.add_experimental_option("prefs", prefs)
        return options

    # Find Chromium/Chrome executable
    chromium_paths = [
        '/snap/bin/chromium',
        '/usr/bin/chromium-browser',
        '/usr/bin/chromium',
        '/usr/bin/google-chrome',
        None  # Let uc auto-detect
    ]

    browser_path = None
    for path in chromium_paths:
        if path is None or os.path.exists(path):
            browser_path = path
            break

    # Try to initialize Chrome driver
    driver = None
    try:
        # First attempt with use_subprocess=True (auto-detect Chrome version)
        options = create_chrome_options(download_dir)
        driver = uc.Chrome(
            options=options,
            browser_executable_path=browser_path,
            use_subprocess=True
        )
    except Exception as e:
        print(f"  [*] First attempt failed: {str(e)[:100]}...")
        try:
            # Try without use_subprocess with FRESH options
            print(f"  [*] Retrying Chrome initialization without use_subprocess...")
            options = create_chrome_options(download_dir)
            driver = uc.Chrome(
                options=options,
                browser_executable_path=browser_path
            )
        except Exception as e2:
            print(f"  [*] Second attempt failed: {str(e2)[:100]}...")
            # Last attempt: let uc find Chrome automatically with minimal config
            print(f"  [*] Final attempt with minimal options...")
            options = create_chrome_options(download_dir)
            driver = uc.Chrome(options=options, use_subprocess=False)

    try:
        # Enable downloads in headless mode using CDP
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": download_dir
        })

        # Visit main page first
        driver.get("https://barttorvik.com/")
        time.sleep(3)

        # Get initial list of CSV files before download
        initial_files = set(f for f in os.listdir(download_dir)
                          if f.startswith('trank_data') and f.endswith('.csv'))

        # Trigger the CSV download with date filters
        url = f"https://barttorvik.com/trank.php?year={year}&begin={begin_date}&end={end_date_full}&csv=1"
        driver.get(url)

        # Wait for download to complete with polling (max 30 seconds)
        print(f"  [*] Waiting for CSV download to complete...")
        latest_file = None
        max_wait = 30
        check_interval = 1
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(check_interval)
            elapsed += check_interval

            # Find new trank_data CSV files
            current_files = set(f for f in os.listdir(download_dir)
                              if f.startswith('trank_data') and f.endswith('.csv'))
            new_files = current_files - initial_files

            if new_files:
                # Found new file(s), get the most recent one
                candidate_files = [os.path.join(download_dir, f) for f in new_files]
                latest_file = max(candidate_files, key=os.path.getctime)

                # Check if file is complete (size not changing)
                initial_size = os.path.getsize(latest_file)
                time.sleep(0.5)
                final_size = os.path.getsize(latest_file)

                if initial_size == final_size and final_size > 0:
                    print(f"  [+] CSV download complete: {os.path.basename(latest_file)}")
                    break

        if not latest_file:
            # If no new file found, try to use the most recent existing one
            all_files = [f for f in os.listdir(download_dir)
                        if f.startswith('trank_data') and f.endswith('.csv')]
            if all_files:
                latest_file = max([os.path.join(download_dir, f) for f in all_files],
                                key=os.path.getctime)
                print(f"  [!] No new file detected, using most recent: {os.path.basename(latest_file)}")
            else:
                print(f"  [-] Error: No trank_data CSV file found in {download_dir}")
                return None
        try:
            # Load CSV with no headers, then assign our headers
            df = pd.read_csv(latest_file, header=None)

            # Assign column names
            num_cols = len(df.columns)
            if num_cols <= len(raw_columns):
                df.columns = raw_columns[:num_cols]
            else:
                # More columns than we have names for
                extra_cols = [f'col_{i}' for i in range(len(raw_columns), num_cols)]
                df.columns = raw_columns + extra_cols

            # Sort by barthag (largest to smallest) to get proper ranking
            df = df.sort_values('barthag', ascending=False).reset_index(drop=True)

            # Format date from year and end_date
            # Need to determine the actual calendar year (not season year)
            if end_date:
                # end_date is in format MMDD
                month = end_date[:2]
                day = end_date[2:4]
                # If Nov/Dec, the calendar year is season_year - 1
                # Otherwise it's the season year
                if int(month) >= 11:  # November or December
                    calendar_year = str(int(year) - 1)
                else:  # January through October
                    calendar_year = year
                date_str = f"{calendar_year}-{month}-{day}"
            else:
                # Default to June 30th of season year
                date_str = f"{year}-06-30"

            # Add date column as first column
            df.insert(0, 'date', date_str)

            # Add rank column (1-indexed)
            df.insert(1, 'rank', range(1, len(df) + 1))

            # Split record (rec) into wins and losses BEFORE dropping it
            # Handle both regular hyphen (-) and en-dash (–)
            df['rec'] = df['rec'].astype(str).str.replace('–', '-')
            df[['wins', 'losses']] = df['rec'].str.split('-', expand=True).astype(int)

            # No renaming needed - keep adjoe and adjde as-is

            # Drop unnecessary columns
            cols_to_drop = [
                'rec',  # Drop original rec column
                'col6',  # Drop total games column
                'col15', 'col20', 'col21', 'col22', 'col23',  # Drop
                'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33',  # Drop dupes
                'col35', 'col36',  # Drop
            ]
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

            # Reorder columns: date, rank, team, barthag, wins, losses, then all stat columns
            stat_cols = [
                'adjoe', 'adjde', 'efg_off_prcnt', 'efg_def_prcnt', 'ftr', 'ftrd', 'tor', 'tord',
                'orb', 'drb', 'adj_t', '2p_prcnt_off', '2p_prcnt_def', '3p_prcnt_off',
                '3p_prcnt_def', '3pr', '3prd', 'wab'
            ]
            col_order = ['date', 'rank', 'team', 'barthag', 'wins', 'losses'] + stat_cols
            df = df[[col for col in col_order if col in df.columns]]

            # Push to database
            success = sqlconn.execute_query(df=df, table_name='leaderboard', if_exists='append')

            # Clean up: Remove only trank_data CSV files in the download directory
            csv_files = [f for f in os.listdir(download_dir) if f.startswith('trank_data') and f.endswith('.csv')]
            for csv_file in csv_files:
                csv_path = os.path.join(download_dir, csv_file)
                try:
                    os.remove(csv_path)
                except Exception as e:
                    print(f"[!] Warning: Could not remove {csv_file}: {e}")

            if success:
                print(f"[+] Data pushed to ncaamb.leaderboard successfully")
            else:
                print(f"[-] Failed to push data to database")

            return df

        except Exception as e:
            print(f"[-] Error processing CSV: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    finally:
        try:
            driver.quit()
        except:
            pass


if __name__ == "__main__":
    from datetime import datetime

    # Get today's date
    today = datetime.now()
    month = f"{today.month:02d}"
    day = f"{today.day:02d}"
    end_date = f"{month}{day}"
    year = '2026'

    # Use project-relative download directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    download_dir = os.path.join(project_dir, 'downloads', 'leaderboard')

    print(f"\nFetching leaderboard for {year} (end date: {month}/{day})...")
    print(f"Download directory: {download_dir}\n")

    result = scrape_barttorvik_csv(
        year=year,
        end_date=end_date,
        output_dir=download_dir
    )

    if result is not None:
        print(f"\n[+] Successfully scraped and pushed {len(result)} teams to database")
    else:
        print(f"\n[-] Failed to scrape leaderboard data")