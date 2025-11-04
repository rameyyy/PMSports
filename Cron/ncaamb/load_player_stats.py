import mysql.connector
from dotenv import load_dotenv
import os
from scrapes.playerstats import scrape_player_stats
import pandas as pd

# Load environment variables
load_dotenv('.env')


def construct_game_id(date_str, team, opponent):
    """
    Construct game_id from date, team, and opponent.
    Game IDs use alphabetically sorted team names.

    Args:
        date_str: Date string in format YYYYMMDD or YYYY-MM-DD
        team: Team name (the team being scraped)
        opponent: Opponent name

    Returns:
        Game ID in format YYYYMMDD_Team1_Team2 (alphabetically sorted)
    """
    # Normalize date format to YYYYMMDD
    if date_str and isinstance(date_str, str):
        date_clean = date_str.replace('-', '')
    else:
        return None

    # Sort team names alphabetically
    teams = sorted([str(team).strip(), str(opponent).strip()])

    game_id = f"{date_clean}_{teams[0]}_{teams[1]}"
    return game_id


def load_player_stats_to_db(year='2025', season=2025):
    """
    Load player stats from Bart Torvik API into player_stats table.

    Args:
        year: Season year (e.g., '2025')
        season: Season number for database (e.g., 2025)
    """
    try:
        # Scrape player stats
        print(f"Loading player stats for {year}...")
        df = scrape_player_stats(year=year)

        if df is None or len(df) == 0:
            print("No player stats data found")
            return

        print(f"\nProcessing {len(df)} player records...")

        # Connect to database
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("NCAAMB_DB"),
        )

        cursor = conn.cursor()

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

        for idx, row in df.iterrows():
            try:
                # Construct game_id
                game_id = construct_game_id(row.get('datetext'), row.get('tt'), row.get('opponent'))

                if not game_id:
                    failed += 1
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

                cursor.execute(insert_query, values)
                successful += 1

                # Print progress every 100 records
                if (successful + failed) % 100 == 0:
                    print(f"Processed {successful + failed}/{len(df)} records...")

            except Exception as e:
                failed += 1
                if failed <= 10:  # Print first 10 errors
                    print(f"Error inserting row {idx}: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        print(f"\n{'='*100}")
        print(f"Load complete: {successful} successful, {failed} failed out of {len(df)} records")
        print(f"{'='*100}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    load_player_stats_to_db(year='2025', season=2025)
