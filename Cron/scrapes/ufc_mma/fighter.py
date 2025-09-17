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

def extract_career_stats(soup):
    stats = {}
    
    # Find the "Career statistics" box
    stat_items = soup.select("ul.b-list__box-list li.b-list__box-list-item")
    
    for item in stat_items:
        label_tag = item.find("i", class_="b-list__box-item-title")
        if label_tag:
            label = label_tag.get_text(strip=True)
            value = label_tag.next_sibling.strip() if label_tag.next_sibling else None
            
            if label and value:
                stats[label] = value
    
    return stats


### Parsing logic END

###

def get_fighter_data_ufc_stats(event_url: str):
    try: 
        response = requests.get(event_url, headers=HEADERS)
        print(response.status_code)
        soup = BeautifulSoup(response.text, "html.parser")
        data = extract_career_stats(soup = soup)
        print(data)
    except Exception as e: print(e)
    # print(response.status_code)
    # soup = BeautifulSoup(response.text, "html.parser")
    # data = extract_career_stats(soup = soup)
    # print(data)

def get_fight_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    fight_links = []
    for a in soup.find_all("a", class_="b-flag"):
        href = a.get("href")
        if href and "fight-details" in href:
            fight_links.append(href)
    
    return fight_links


###

def get_fighter_data(event_url: str):
    response = requests.get(event_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    link=get_ufcstats_link(soup_or_html=soup)
    if link:
        print(link)
        get_fighter_data_ufc_stats(link)
    else:
        print('notwork')
    #fighters_dob = get_fighter_dob(soup)
    #print(event_url)
    # print(link)
    # response = requests.get(link, headers=HEADERS)
    # print(response.status_code)
    # print(fighters_dob)