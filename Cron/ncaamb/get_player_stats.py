import mysql.connector
from dotenv import load_dotenv
import os
from scrapes.playerstats import scrape_player_stats
import pandas as pd

# Load environment variables
load_dotenv('.env')


def construct_game_id(date_str, team, opponent, year=2025):
    """
    Construct game_id from date, team, and opponent.
    Game IDs use alphabetically sorted team names.

    Args:
        date_str: Date string in format MM-DD or YYYYMMDD or YYYY-MM-DD
        team: Team name (the team being scraped)
        opponent: Opponent name
        year: The season year (e.g., 2025 for 2024-2025 season)

    Returns:
        Game ID in format YYYYMMDD_Team1_Team2 (alphabetically sorted)
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    # If datetext is just MM-DD (like "11-6"), we need to figure out the year
    # The 2024-2025 season starts in Nov 2024 and ends in Mar/Apr 2025
    if len(date_str) <= 5:  # MM-DD or M-D format
        parts = date_str.split('-')
        if len(parts) == 2:
            try:
                month = int(parts[0])
                day = int(parts[1])

                # November-December = previous year, January-April = current year
                if month >= 11:
                    actual_year = year - 1
                else:
                    actual_year = year

                date_clean = f"{actual_year}{month:02d}{day:02d}"
            except ValueError:
                return None
        else:
            return None
    else:
        # Already has year, just clean it up
        date_clean = date_str.replace('-', '')

    # Sort team names alphabetically
    teams = sorted([str(team).strip(), str(opponent).strip()])

    game_id = f"{date_clean}_{teams[0]}_{teams[1]}"
    return game_id


def load_player_stats_to_db(year='2025', season=2025, skip_missing_games=True):
    """
    Load player stats from Bart Torvik API into player_stats table.

    Args:
        year: Season year (e.g., '2025')
        season: Season number for database (e.g., 2025)
        skip_missing_games: If True, skip player stats for games not in DB (default: True)
    """
    try:
        # Scrape player stats
        print(f"Loading player stats for {year}...")
        df = scrape_player_stats(year=year)

        if df is None or len(df) == 0:
            print("No player stats data found")
            return

        print(f"\nProcessing {len(df)} player records...")

        # Debug: print first few rows to see data format
        print("\nFirst 5 rows of data:")
        print(df[['datetext', 'tt', 'opponent']].head())

        # Debug: construct first few game_ids and see what they look like
        print("\nFirst 5 constructed game_ids:")
        for i in range(min(5, len(df))):
            gid = construct_game_id(df.iloc[i]['datetext'], df.iloc[i]['tt'], df.iloc[i]['opponent'], year=int(year))
            print(f"  Row {i}: {gid}")

        # Connect to database
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("NCAAMB_DB"),
        )

        cursor = conn.cursor()

        # Debug: check if first few game_ids exist in games table
        print("\nChecking if first 5 game_ids exist in games table:")
        for i in range(min(5, len(df))):
            gid = construct_game_id(df.iloc[i]['datetext'], df.iloc[i]['tt'], df.iloc[i]['opponent'], year=int(year))
            cursor.execute("SELECT COUNT(*) FROM games WHERE game_id = %s", (gid,))
            count = cursor.fetchone()[0]
            exists = "EXISTS" if count > 0 else "MISSING"
            print(f"  {gid}: {exists}")

        # Pre-load all existing game_ids into a set for fast lookup
        print("\nLoading existing game_ids from database...")
        cursor.execute(f"SELECT game_id FROM games WHERE season = {season}")
        existing_games = set(row[0] for row in cursor.fetchall())
        print(f"Found {len(existing_games)} games in database for season {season}")

        print("\n" + "="*100)

        # Prepare insert statement
        insert_query = """
        INSERT INTO player_stats (
            game_id, season, player_name, player_id, team, opponent,
            numdate, datetext, opstyle, quality, win1, muid, win2, Min_per,
            ORtg, useage, eFG, TS_per, ORB_per, DRB_per, AST_per, TO_per,
            dunksmade, dunksatt, rimmade, rimatt, midmade, midatt, twoPM, twoPA,
            TPM, TPA, FTM, FTA, bpm_rd, Obpm, Dbpm, bpm_net, pts, ORB, DRB,
            AST, TOV, STL, BLK, stl_per, blk_per, PF, possessions, bpm, sbpm,
            loc, inches, cls
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s
        ) ON DUPLICATE KEY UPDATE
            Min_per = VALUES(Min_per),
            ORtg = VALUES(ORtg),
            useage = VALUES(useage),
            eFG = VALUES(eFG),
            TS_per = VALUES(TS_per),
            ORB_per = VALUES(ORB_per),
            DRB_per = VALUES(DRB_per),
            AST_per = VALUES(AST_per),
            TO_per = VALUES(TO_per),
            pts = VALUES(pts),
            ORB = VALUES(ORB),
            DRB = VALUES(DRB),
            AST = VALUES(AST),
            TOV = VALUES(TOV),
            STL = VALUES(STL),
            BLK = VALUES(BLK)
        """

        successful = 0
        failed = 0
        skipped_missing_games = 0
        batch_size = 10000
        batch_values = []
        missing_games = set()

        start_time = pd.Timestamp.now()

        for idx, row in df.iterrows():
            try:
                # Construct game_id
                game_id = construct_game_id(row.get('datetext'), row.get('tt'), row.get('opponent'), year=int(year))

                if not game_id:
                    failed += 1
                    continue

                # Check if game exists in database
                if game_id not in existing_games:
                    if skip_missing_games:
                        skipped_missing_games += 1
                        missing_games.add(game_id)
                        continue
                    else:
                        failed += 1
                        if failed <= 10:
                            print(f"Error: Game not found in database: {game_id}")
                        continue

                # Prepare values - convert NaN to None for SQL NULL
                # Note: DataFrame has 'Usage' but table has 'useage' (reserved keyword)
                values = (
                    game_id, season, row.get('pp'), row.get('pid'), row.get('tt'), row.get('opponent'),
                    row.get('numdate'), row.get('datetext'), row.get('opstyle'), row.get('quality'),
                    row.get('win1'), row.get('muid'), row.get('win2'), row.get('Min_per'),
                    row.get('ORtg'), row.get('Usage'), row.get('eFG'), row.get('TS_per'),
                    row.get('ORB_per'), row.get('DRB_per'), row.get('AST_per'), row.get('TO_per'),
                    row.get('dunksmade'), row.get('dunksatt'), row.get('rimmade'), row.get('rimatt'),
                    row.get('midmade'), row.get('midatt'), row.get('twoPM'), row.get('twoPA'),
                    row.get('TPM'), row.get('TPA'), row.get('FTM'), row.get('FTA'),
                    row.get('bpm_rd'), row.get('Obpm'), row.get('Dbpm'), row.get('bpm_net'),
                    row.get('pts'), row.get('ORB'), row.get('DRB'),
                    row.get('AST'), row.get('TOV'), row.get('STL'), row.get('BLK'),
                    row.get('stl_per'), row.get('blk_per'), row.get('PF'),
                    row.get('possessions'), row.get('bpm'), row.get('sbpm'),
                    row.get('loc'), row.get('inches'), row.get('cls')
                )

                # Convert NaN to None for NULL values
                values = tuple(None if (isinstance(v, float) and pd.isna(v)) else v for v in values)

                batch_values.append(values)
                successful += 1

                # Commit every batch_size records
                if successful % batch_size == 0:
                    cursor.executemany(insert_query, batch_values)
                    conn.commit()
                    batch_values = []

                    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                    rate = successful / elapsed if elapsed > 0 else 0
                    eta_remaining = (len(df) - successful) / rate if rate > 0 else 0
                    print(f"Processed {successful}/{len(df)} records ({rate:.0f} rec/sec, ETA: {eta_remaining/60:.1f} min)...")

            except Exception as e:
                failed += 1
                if failed <= 10:  # Print first 10 errors
                    print(f"Error processing row {idx}: {e}")

        # Insert remaining records
        if batch_values:
            cursor.executemany(insert_query, batch_values)
            conn.commit()

        cursor.close()
        conn.close()

        print(f"\n{'='*100}")
        print(f"Load complete:")
        print(f"  - Successful: {successful}")
        print(f"  - Failed: {failed}")
        print(f"  - Skipped (missing games): {skipped_missing_games}")
        print(f"  - Total processed: {len(df)}")

        if missing_games:
            print(f"\n⚠️  WARNING: {len(missing_games)} unique games not found in database!")
            print(f"This means you need to load game data for season {season} first.")
            print(f"\nSample of missing game_ids (first 10):")
            for game_id in sorted(list(missing_games))[:10]:
                print(f"  - {game_id}")
            if len(missing_games) > 10:
                print(f"  ... and {len(missing_games) - 10} more")
            print(f"\nTo fix this:")
            print(f"  1. Run: python main.py  (with year='{year}' and season={season})")
            print(f"  2. Then re-run this script")

        print(f"{'='*100}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    load_player_stats_to_db(year='2020', season=2020)
