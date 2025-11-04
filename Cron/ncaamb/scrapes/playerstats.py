import requests
import gzip
import json
import pandas as pd
from io import BytesIO

# Column headers for player stats
PLAYER_STATS_COLUMNS = [
    'numdate', 'datetext', 'opstyle', 'quality', 'win1', 'opponent', 'muid', 'win2', 'Min_per',
    'ORtg', 'Usage', 'eFG', 'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per',
    'dunksmade', 'dunksatt', 'rimmade', 'rimatt', 'midmade', 'midatt', 'twoPM', 'twoPA',
    'TPM', 'TPA', 'FTM', 'FTA', 'bpm_rd', 'Obpm', 'Dbpm', 'bpm_net', 'pts', 'ORB', 'DRB',
    'AST', 'TOV', 'STL', 'BLK', 'stl_per', 'blk_per', 'PF', 'possessions', 'bpm', 'sbpm',
    'loc', 'tt', 'pp', 'inches', 'cls', 'pid', 'year'
]

def scrape_player_stats(year):
    """
    Scrape player stats for all games from Bart Torvik's advanced games JSON

    Args:
        year: Season year (e.g., '2024')

    Returns:
        DataFrame with player stats or None if failed
    """
    url = f"https://barttorvik.com/{year}_all_advgames.json.gz"

    try:
        print(f"Fetching player stats for {year}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Try to decompress gzip data, fallback to raw JSON if it's not gzipped
        print("Decompressing data...")
        try:
            with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz_file:
                data = json.loads(gz_file.read().decode('utf-8'))
        except (gzip.BadGzipFile, EOFError):
            # File is not gzipped, parse as raw JSON
            print("File is not gzipped, parsing as raw JSON...")
            data = json.loads(response.text)

        print(f"Successfully loaded data from {year}")

        # Convert to DataFrame
        if isinstance(data, list):
            print(f"Total records: {len(data)}")
            df = pd.DataFrame(data)
            df.columns = PLAYER_STATS_COLUMNS
            print(f"\nDataFrame shape: {df.shape}")
            print(f"\nFirst few rows:")
            print(df.head())
            return df
        else:
            print(f"Unexpected data type: {type(data)}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    data = scrape_player_stats(year='2024')
