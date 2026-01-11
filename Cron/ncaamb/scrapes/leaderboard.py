import undetected_chromedriver as uc
import pandas as pd
import time
import os
from . import sqlconn

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
    
    # Setup download directory
    download_dir = os.path.abspath(output_dir)

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

    options = uc.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')

    # Set download preferences
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    options.add_experimental_option("prefs", prefs)

    try:
        # Try to initialize Chrome driver with common configurations
        driver = uc.Chrome(
            options=options,
            use_subprocess=True,
            driver_executable_path=None,
            browser_executable_path=None
        )
    except Exception as e:
        # If that fails, try without use_subprocess
        print(f"  [*] Retrying Chrome initialization without use_subprocess...")
        driver = uc.Chrome(options=options)

    try:
        # Visit main page first
        driver.get("https://barttorvik.com/")
        time.sleep(3)

        # Trigger the CSV download with date filters
        url = f"https://barttorvik.com/trank.php?year={year}&begin={begin_date}&end={end_date_full}&csv=1"
        driver.get(url)

        # Wait for download to complete
        time.sleep(5)

        # Find the downloaded file
        downloaded_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]

        if not downloaded_files:
            print("❌ Error: No CSV file found in download directory")
            return None

        # Get the most recent CSV
        latest_file = max([os.path.join(download_dir, f) for f in downloaded_files],
                        key=os.path.getctime)
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
                    print(f"⚠️  Warning: Could not remove {csv_file}: {e}")

            if success:
                print(f"✅ Data pushed to ncaamb.leaderboard successfully")
            else:
                print(f"❌ Failed to push data to database")

            return df

        except Exception as e:
            print(f"❌ Error processing CSV: {e}")
            return None
            
    finally:
        try:
            driver.quit()
        except:
            pass