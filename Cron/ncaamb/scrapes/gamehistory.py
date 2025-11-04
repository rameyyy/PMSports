import requests
import pandas as pd
from . import sqlconn
import json
from datetime import datetime
from urllib.parse import quote

def scrape_game_history(year='2025', team=None):
    """
    Scrape game history for a specific team from Bart Torvik's API

    Args:
        year: Season year (e.g., '2025')
        team: Team name (e.g., 'Auburn')

    Returns:
        DataFrame with game history data or None if failed
    """
    if not team:
        print("Error: team parameter is required")
        return None

    # URL-encode team name to handle special characters like &
    encoded_team = quote(team, safe='')
    url = f"https://barttorvik.com/getgamestats.php?year={year}&tvalue={encoded_team}"

    try:
        print(f"Fetching game history for {team} ({year})...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = json.loads(response.text)

        if not data or len(data) == 0:
            print(f"No game data found for {team}")
            return None

        # Process the data
        df = pd.DataFrame(data)

        # Expand column 29 which contains nested JSON string
        df[29] = df[29].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        # Flatten the nested list in column 29
        expanded = df[29].apply(lambda x: pd.Series(x) if isinstance(x, list) else x)

        # Rename expanded columns to avoid conflicts (start from col 31+)
        expanded.columns = [f'nested_{i}' for i in range(len(expanded.columns))]

        # Drop the original nested column and concatenate
        df = df.drop(columns=[29])
        df = pd.concat([df, expanded], axis=1)

        # Now transform the dataframe according to specifications
        df = transform_game_history(df, year)

        print(f"Retrieved {len(df)} games for {team}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def transform_game_history(df, year):
    """Transform raw game history data to proper format"""

    # Column 0: Date - reformat to SQL date format
    df['date'] = pd.to_datetime(df[0], format='%m/%d/%y').dt.strftime('%Y-%m-%d')

    # Column 1: Game type (0=Regular Non-Conference, 1=Regular Conference, 2=Conference Tournament, 3=NCAA Tournament)
    game_type_map = {0: 'Regular Season Non-Conference', 1: 'Regular Season Conference',
                     2: 'Conference Tournament', 3: 'NCAA Tournament'}
    df['game_type'] = df[1].apply(lambda x: game_type_map.get(x, 'Unknown'))

    # Column 2: Team name
    df['team'] = df[2]

    # Column 3: Team conference
    df['conference'] = df[3]

    # Column 4: Opponent team name
    df['opponent'] = df[4]

    # Column 5: H/A/N (Home/Away/Neutral) - convert to team names or N
    def convert_location(row_location, team_name, opponent_name):
        if row_location == 'H':
            return team_name
        elif row_location == 'A':
            return opponent_name
        else:  # 'N' for neutral
            return 'N'

    df['location'] = df.apply(lambda row: convert_location(row[5], row[2], row[4]), axis=1)

    # Column 6: Parse Win/Loss and scores
    def parse_result(result_str):
        if 'W' in result_str:
            win_loss = 1
            scores = result_str.replace('W, ', '').split('-')
            team_score = int(scores[0])  # Team's score (the winner's)
            opp_score = int(scores[1])   # Opponent's score
        else:
            win_loss = 0
            scores = result_str.replace('L, ', '').split('-')
            team_score = int(scores[1])  # Team's score (the loser's, second number)
            opp_score = int(scores[0])   # Opponent's score (the winner's, first number)

        return win_loss, team_score, opp_score

    df[['win_loss', 'team_score', 'opp_score']] = df[6].apply(lambda x: pd.Series(parse_result(x)))

    # Columns 7-9: Team stats (adjoe, adjde, eff)
    df['team_adjoe'] = df[7]
    df['team_adjde'] = df[8]
    df['team_eff'] = df[9]

    # Columns 10-13: Team shooting stats (efg%, to%, or%, ftr)
    df['team_efg_pct'] = df[10]
    df['team_to_pct'] = df[11]
    df['team_or_pct'] = df[12]
    df['team_ftr'] = df[13]

    # Columns 14-18: Team defense stats (eff, efg%, to%, or%, ftr, g_sc)
    df['team_def_eff'] = df[14]
    df['team_def_efg_pct'] = df[15]
    df['team_def_to_pct'] = df[16]
    df['team_def_or_pct'] = df[17]
    df['team_def_ftr'] = df[18]

    # Column 19: Team game score
    df['team_g_sc'] = df[19]

    # Column 20: Opponent conference
    df['opp_conference'] = df[20]

    # Column 27: Plus/Minus
    df['plus_minus'] = df[27]

    # OT determination from nested data (position 1 is 200/OT value)
    df['ot'] = df['nested_1'].apply(lambda x: 0 if x == 200 else 1)

    # Sort by date to ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)

    # Assign coaches AFTER sorting to maintain proper game associations
    # Columns 25-26: Coaches
    df['team_coach'] = df[25]
    df['opp_coach'] = df[26]

    # Calculate record (wins-losses) as cumulative sum
    df['wins'] = df['win_loss'].cumsum()
    df['losses'] = (1 - df['win_loss']).cumsum()
    df['record'] = df['wins'].astype(str) + '-' + df['losses'].astype(str)

    # Extract detailed box score stats from nested data
    # Nested array format: [date, 200/OT, opp_name, team_name,
    #                       opp_FGM, opp_FGA, opp_3PM, opp_3PA, opp_FTM, opp_FTA, opp_OREB, opp_DREB, opp_TREB, opp_AST, opp_TO, opp_STL, opp_BLK, opp_PF, opp_PTS,
    #                       team_FGM, team_FGA, team_3PM, team_3PA, team_FTM, team_FTA, team_OREB, team_DREB, team_TREB, team_AST, team_TO, team_STL, team_BLK, team_PF, team_PTS, ...]

    # Opponent stats start at nested_4 (after date, 200/OT, opp_name, team_name)
    # Team stats start at nested_19 (15 stats per team)

    # Team shooting stats from nested data (indices 4-18)
    # Order: FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, TREB, AST, STL, BLK, TO, PF, PTS
    df['team_fgm'] = df['nested_4'].astype(int)
    df['team_fga'] = df['nested_5'].astype(int)
    df['team_3pm'] = df['nested_6'].astype(int)
    df['team_3pa'] = df['nested_7'].astype(int)
    df['team_2pm'] = (df['nested_4'] - df['nested_6']).astype(int)  # FGM - 3PM
    df['team_2pa'] = (df['nested_5'] - df['nested_7']).astype(int)  # FGA - 3PA
    df['team_ftm'] = df['nested_8'].astype(int)
    df['team_fta'] = df['nested_9'].astype(int)
    df['team_oreb'] = df['nested_10'].astype(int)
    df['team_dreb'] = df['nested_11'].astype(int)
    df['team_treb'] = df['nested_12'].astype(int)
    df['team_ast'] = df['nested_13'].astype(int)
    df['team_stl'] = df['nested_14'].astype(int)
    df['team_blk'] = df['nested_15'].astype(int)
    df['team_to'] = df['nested_16'].astype(int)
    df['team_pf'] = df['nested_17'].astype(int)

    # Opponent shooting stats from nested data (indices 19-33)
    # Order: FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, TREB, AST, STL, BLK, TO, PF, PTS
    df['opp_fgm'] = df['nested_19'].astype(int)
    df['opp_fga'] = df['nested_20'].astype(int)
    df['opp_3pm'] = df['nested_21'].astype(int)
    df['opp_3pa'] = df['nested_22'].astype(int)
    df['opp_2pm'] = (df['nested_19'] - df['nested_21']).astype(int)  # FGM - 3PM
    df['opp_2pa'] = (df['nested_20'] - df['nested_22']).astype(int)  # FGA - 3PA
    df['opp_ftm'] = df['nested_23'].astype(int)
    df['opp_fta'] = df['nested_24'].astype(int)
    df['opp_oreb'] = df['nested_25'].astype(int)
    df['opp_dreb'] = df['nested_26'].astype(int)
    df['opp_treb'] = df['nested_27'].astype(int)
    df['opp_ast'] = df['nested_28'].astype(int)
    df['opp_stl'] = df['nested_29'].astype(int)
    df['opp_blk'] = df['nested_30'].astype(int)
    df['opp_to'] = df['nested_31'].astype(int)
    df['opp_pf'] = df['nested_32'].astype(int)

    # Select and rename final columns
    final_columns = [
        'date', 'game_type', 'team', 'conference', 'opponent', 'location',
        'win_loss', 'record', 'team_score', 'opp_score',
        'team_adjoe', 'team_adjde', 'team_eff',
        'team_efg_pct', 'team_to_pct', 'team_or_pct', 'team_ftr',
        'team_def_eff', 'team_def_efg_pct', 'team_def_to_pct', 'team_def_or_pct', 'team_def_ftr', 'team_g_sc', 'opp_conference',
        'team_fgm', 'team_fga', 'team_2pm', 'team_2pa', 'team_3pm', 'team_3pa',
        'team_ftm', 'team_fta', 'team_oreb', 'team_dreb', 'team_treb',
        'team_ast', 'team_to', 'team_stl', 'team_blk', 'team_pf',
        'opp_fgm', 'opp_fga', 'opp_2pm', 'opp_2pa', 'opp_3pm', 'opp_3pa',
        'opp_ftm', 'opp_fta', 'opp_oreb', 'opp_dreb', 'opp_treb',
        'opp_ast', 'opp_to', 'opp_stl', 'opp_blk', 'opp_pf',
        'team_coach', 'opp_coach', 'plus_minus', 'ot'
    ]

    df = df[final_columns]

    return df


def scrape_all_teams(year='2025', season=2025):
    """
    Scrape game history for all teams in the database and insert into MySQL

    Args:
        year: Season year for scraping (e.g., '2025')
        season: Season year for database insertion (e.g., 2025)
    """
    # Get all distinct teams from leaderboard
    conn = sqlconn.create_connection()
    if not conn:
        print("Could not connect to database")
        return

    query = "SELECT DISTINCT team FROM leaderboard ORDER BY team"
    results = sqlconn.fetch(conn, query)
    conn.close()

    if not results:
        print("No teams found in database")
        return

    teams = [row['team'] for row in results]
    total_teams = len(teams)

    print(f"Starting game history scrape for {total_teams} teams")
    print(f"Year: {year}, Season: {season}")
    print("-" * 60)

    successful = 0
    failed = 0

    for idx, team in enumerate(teams, 1):
        percentage = (idx / total_teams) * 100
        print(f"\n[{idx}/{total_teams}] ({percentage:.1f}%) Processing: {team}")

        try:
            df = scrape_game_history(year=year, team=team)
            if df is not None and len(df) > 0:
                print(f"Success - {len(df)} games retrieved")

                # Insert team into teams table (only this team, not opponents)
                sqlconn.push_to_teams(df, season)

                # Insert games into games table
                sqlconn.push_to_games(df, season, year)

                successful += 1
            else:
                print(f"No games found")
                failed += 1
        except Exception as e:
            print(f"Error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Game history scrape complete!")
    print(f"Successful: {successful}/{total_teams}")
    print(f"Failed: {failed}/{total_teams}")
    print("=" * 60)
