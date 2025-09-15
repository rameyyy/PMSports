import requests
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL = "https://www.tapology.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

### Parsing logic START

def get_fighter_dob(soup: BeautifulSoup):
    dob_label = soup.find("strong", string="| Date of Birth:")
    if not dob_label:
        return None
    dob_str = dob_label.find_next("span").get_text(strip=True)
    return datetime.strptime(dob_str, "%Y %b %d").date()

### Parsing logic END

def get_fighter_data(event_url: str):
    response = requests.get(event_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    fighters_dob = get_fighter_dob(soup)
    print(response.status_code)
    print(fighters_dob)