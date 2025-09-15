import requests
from bs4 import BeautifulSoup
from .utils import get_event_title, get_event_date_location, ufc_weight_class
import re
from datetime import datetime

BASE_URL = "https://www.tapology.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def parse_fight_details(tag):
    records_to_find = tag.find_all(lambda tag: tag.name == 'span' and 'text-[15px]' in tag.get('class', []))
    fighter_links = tag.find_all('a', class_='link-primary-red')
    cardtypes = tag.find_all('a', class_='hover:text-neutral-950')
    cardtypes_arr = [cardtype.get_text(strip=True) for cardtype in cardtypes]
    cardtype = cardtypes_arr[0]
    elements = tag.find_all('td', class_='text-neutral-950')
    data = str(elements)
    def extract_age(html_str):
        """
        Extracts age in the format 'XX years, XX months, XX weeks, XX days' from a string.
        """
        # Regex to match patterns like '31 years, 3 months, 2 weeks, 1 day'
        
        pattern = r'\d+\s+years?(?:,\s*\d+\s+months?)?(?:,\s*\d+\s+weeks?)?(?:,\s*\d+\s+days?)?'

        matches = re.findall(pattern, html_str)

        # Filter: keep only matches with 3 or more time units
        filtered = [m for m in matches if len(re.findall(r'\d+\s+(?:years?|months?|weeks?|days?)', m)) >= 3]

        # Remove duplicates and get first two
        unique = list(dict.fromkeys(filtered))[:2]
        if len(unique) == 0:
            unique.append('')
            unique.append('')
        elif len(unique) == 1:
            unique.append('')
        if unique:
            return unique
        return None
    age_of_fighter = extract_age(data)
    def extract_unique_fighters(a_tags):
        seen = set()
        fighters = []
        for tag in a_tags:
            name = tag.text.strip()
            link_suf = tag.get('href', '').strip()
            link = f'{BASE_URL}{link_suf}'
            key = (name, link)
            if key not in seen:
                seen.add(key)
                fighters.append([name, link])
            if len(fighters) == 2:
                break
        return fighters

    def get_fighter_records(tag_list):
        """
        Extracts fighter records from a list of BeautifulSoup tags.
        """
        return [
            tag.get_text(strip=True)
            for tag in tag_list
            if 'order-1' in tag.get('class', []) or 'order-2' in tag.get('class', [])
        ]
        
    records = get_fighter_records(records_to_find)
    if records is None or len(records) < 2:
        records = ['', '']
    fighters_data = extract_unique_fighters(fighter_links)
    def extract_mobile_text(cell):
        mobile_div = cell.select_one('div.md\\:hidden')
        return mobile_div.get_text(separator=' ', strip=True) if mobile_div else cell.get_text(strip=True)
    
    def get_fighter_image_url(soup: BeautifulSoup, fighter_name: str) -> str | None:
        img_tag = soup.find("img", alt=lambda x: x and fighter_name.lower() in x.lower())
        return img_tag["src"] if img_tag else None

    def extract_column_data(rows, label, extract_age=False):
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 5 and label.lower() in cols[2].text.strip().lower():
                if extract_age:
                    span1 = cols[0].find('span')
                    span2 = cols[4].find('span')
                    age1 = span1.text.strip() if span1 else ''
                    age2 = span2.text.strip() if span2 else ''
                    return age1, age2
                return extract_mobile_text(cols[0]), extract_mobile_text(cols[4])
        return '', ''

    # 🟨 Fight Overview
    weight_class = tag.select_one('span.bg-tap_darkgold')
    weight_class_weight = weight_class.text.strip() if weight_class else ''
    fight_overview = {
        'fight_card_type': cardtype,
        'weight_class_weight': weight_class_weight,
        'weight_class': ufc_weight_class(weight_class_weight)
    }

    fighter1_name = fighters_data[0][0]
    fighter2_name = fighters_data[1][0]
    fighter1_link = fighters_data[0][1]
    fighter2_link = fighters_data[1][1]
    fighter1_record = records[0]
    fighter2_record = records[1]

    # 📊 Fighter Stats Table
    table = tag.select_one('table#boutComparisonTable')
    rows = table.find_all('tr') if table else []

    nickname1, nickname2 = extract_column_data(rows, 'Nickname')
    odds1, odds2 = extract_column_data(rows, 'Betting Odds')
    age1, age2 = age_of_fighter[0], age_of_fighter[1]
    weight1, weight2 = extract_column_data(rows, 'Latest Weight')
    height1, height2 = extract_column_data(rows, 'Height')
    reach1, reach2 = extract_column_data(rows, 'Reach')
    fighter1_img_url = get_fighter_image_url(tag, fighter1_name)
    fighter2_img_url = get_fighter_image_url(tag, fighter2_name)

    return {
        'fight_overview': fight_overview,
        'fighter1': {
            'fighter_name': fighter1_name,
            'record': fighter1_record,
            'nickname': nickname1,
            'betting_odds': odds1,
            'age_at_fight': age1,
            'latest_weight': weight1,
            'height': height1,
            'reach': reach1,
            'link': fighter1_link,
            'img_link': fighter1_img_url
        },
        'fighter2': {
            'fighter_name': fighter2_name,
            'record': fighter2_record,
            'nickname': nickname2,
            'betting_odds': odds2,
            'age_at_fight': age2,
            'latest_weight': weight2,
            'height': height2,
            'reach': reach2,
            'link': fighter2_link,
            'img_link': fighter2_img_url
        }
    }

def is_future_event(date_str: str) -> bool:
    # Parse only the date portion
    dt = datetime.strptime(date_str, "%A %m.%d.%Y at %I:%M %p ET")
    today = datetime.today()
    
    # Compare by date only (ignore time)
    return dt.date() >= today.date()

def get_all_events(group: str = "ufc", past: bool = False, page: int = 1):
    if past:
        fightcenter_url = f"{BASE_URL}/fightcenter?group={group}&schedule=results&page={page}"
    else:
        fightcenter_url = f"{BASE_URL}/fightcenter?group={group}&schedule=upcoming"
    response = requests.get(fightcenter_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    with open("t_p1.txt", "w", encoding="utf-8") as f:
        f.write(soup.prettify())   # prettify makes it formatted nicely


    urls = []
    for tag in soup.select('a[href^="/fightcenter/events/"]'):
        url = f"{BASE_URL}{tag['href']}"
        if url not in urls and group in url:
            urls.append(url)
    return urls

def get_event_data(event_url: str, getting_old_data: bool):
    response = requests.get(event_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    
    title = get_event_title(soup)
    date, location = get_event_date_location(soup)

    def get_data():
        fights = []
        section_cards = soup.select('div[id^="sectionFightCard"]')
        for section in section_cards:
            fight_tags = section.select(
                'li.border-b.border-dotted.border-tap_6[data-controller="table-row-background"]'
            )
            for fight in fight_tags:
                fights.append(parse_fight_details(fight))

        return {
            "url": event_url,
            "title": title,
            "date": date,
            "location": location,
            "fights": fights,
        }
    if date is None:
        return
    is_future_event_bool = is_future_event(date)
    if getting_old_data == False and is_future_event_bool == True:
        return get_data()
    if getting_old_data == False and is_future_event_bool == False:
        return
    if getting_old_data == True and is_future_event_bool == True:
        return
    return get_data()
        