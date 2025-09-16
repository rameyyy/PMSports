import requests
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL = "https://www.tapology.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

### Grab ufc-stats link from Tapology ufc fighter stats page response soup:
def get_ufcstats_link(soup_or_html) -> str | None:
    # Accept HTML string or BeautifulSoup
    soup = BeautifulSoup(soup_or_html, "html.parser") if isinstance(soup_or_html, str) else soup_or_html

    # 1) Prefer the "Resource Links:" section if present
    label = soup.find("strong", string=lambda s: s and s.strip().lower() == "resource links:")
    if label:
        container = label.find_parent().find_next("div") or label.find_next("div")
        if container:
            a = container.find("a", href=lambda h: h and "ufcstats.com/fighter-details" in h)
            if a:
                return a["href"]

    # 2) Fallback: search anywhere on the page
    a = soup.find("a", href=lambda h: h and "ufcstats.com/fighter-details" in h)
    return a["href"] if a else None

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
    link=get_ufcstats_link(soup_or_html=soup)
    fighters_dob = get_fighter_dob(soup)
    print(event_url)
    print(link)
    response = requests.get(link, headers=HEADERS)
    print(response.status_code)
    # print(fighters_dob)