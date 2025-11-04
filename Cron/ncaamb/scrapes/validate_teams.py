import requests
import gzip
import json
from urllib.parse import quote
from io import BytesIO

def get_unique_teams(year='2025'):
    """
    Extract all unique team names from player stats data
    """
    print(f"Fetching player stats for {year}...")
    url = f"https://barttorvik.com/{year}_all_advgames.json.gz"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Parse JSON data (file is not actually gzipped despite .gz extension)
        data = json.loads(response.text)

        # Extract unique teams from 'tt' column (this team)
        # Data is a list of lists, 'tt' (this team) is at index 47
        teams = set()
        for record in data:
            if isinstance(record, list) and len(record) > 47:
                team = record[47]  # tt (this team)
                if team and isinstance(team, str):
                    teams.add(team)

        return sorted(list(teams))

    except Exception as e:
        print(f"Error fetching player stats: {e}")
        return []


def test_team_scrape(team, year='2025'):
    """
    Test if a team can be scraped from Bart Torvik API
    """
    # Convert team to string in case it's not
    team = str(team).strip()
    encoded_team = quote(team, safe='')
    url = f"https://barttorvik.com/getgamestats.php?year={year}&tvalue={encoded_team}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = json.loads(response.text)

        if not data or len(data) == 0:
            return False, "No data returned"

        return True, len(data)

    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except json.JSONDecodeError:
        return False, "Invalid JSON response"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("Extracting unique teams from player stats...\n")
    teams = get_unique_teams(year='2025')

    print(f"Found {len(teams)} unique teams\n")
    print("=" * 100)
    print(f"{'Team':<40} {'Status':<15} {'Games/Error':<50}")
    print("=" * 100)

    successful = 0
    failed = 0
    failed_teams = []

    for team in teams:
        success, result = test_team_scrape(team)

        if success:
            print(f"{team:<40} {'SUCCESS':<15} {result} games")
            successful += 1
        else:
            print(f"{team:<40} {'FAILED':<15} {result}")
            failed += 1
            failed_teams.append((team, result))

    print("=" * 100)
    print(f"\nSummary: {successful} successful, {failed} failed out of {len(teams)} teams\n")

    if failed_teams:
        print("Failed Teams:")
        print("-" * 100)
        for team, error in failed_teams:
            print(f"  {team:<40} → {error}")
