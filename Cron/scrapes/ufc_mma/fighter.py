import requests
from bs4 import BeautifulSoup, NavigableString
from datetime import datetime
import re

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

def clean_fighter_stats(raw: dict) -> dict:
    """
    Normalize fighter stats from a raw dict into clean formats.
    
    Args:
        raw (dict): Raw fighter stats with keys like 'Height:', 'Weight:', etc.
    
    Returns:
        dict: Cleaned fighter stats with consistent keys and value types.
    """
    cleaned = {}

    try:
        # Height: '6\' 1"' -> inches
        if 'Height:' in raw:
            try:
                feet, inches = raw['Height:'].replace('"', '').split("'")
                total_inches = int(feet.strip()) * 12 + int(inches.strip())
                cleaned['height'] = total_inches
            except Exception:
                cleaned['height'] = None
    except Exception:
        cleaned['height'] = None

    try:
        # Weight: '185 lbs.' -> int
        if 'Weight:' in raw:
            cleaned['weight'] = int(raw['Weight:'].split()[0])
    except Exception:
        cleaned['weight'] = None

    try:
        # Reach: '74"' -> int
        if 'Reach:' in raw:
            cleaned['reach'] = int(raw['Reach:'].replace('"', '').strip())
    except Exception:
        cleaned['reach'] = None

    try:
        # Stance: str
        if 'STANCE:' in raw:
            cleaned['stance'] = raw['STANCE:'].strip()
    except Exception:
        cleaned['stance'] = None

    try:
        # DOB: 'Mar 28, 1988' -> MM-DD-YYYY
        if 'DOB:' in raw:
            dt = datetime.strptime(raw['DOB:'], "%b %d, %Y")
            cleaned['dob'] = dt.strftime("%m-%d-%Y")
    except Exception:
        cleaned['dob'] = None

    # Convert percentage strings like '58%' safely
    def pct_to_float(val):
        try:
            return float(val.replace('%', '').strip()) / 100.0
        except Exception:
            return None

    # Convert numeric safely
    def to_float(val):
        try:
            return float(val.strip())
        except Exception:
            return None

    # Map other keys
    mapping = {
        'SLpM:': ('slpm', to_float),
        'Str. Acc.:': ('str_acc', pct_to_float),
        'SApM:': ('sapm', to_float),
        'Str. Def:': ('str_def', pct_to_float),
        'TD Avg.:': ('td_avg', to_float),
        'TD Acc.:': ('td_acc', pct_to_float),
        'TD Def.:': ('td_def', pct_to_float),
        'Sub. Avg.:': ('sub_avg', to_float),
    }

    for old_key, (new_key, func) in mapping.items():
        try:
            if old_key in raw:
                cleaned[new_key] = func(raw[old_key])
        except Exception:
            cleaned[new_key] = None

    return cleaned

### Parsing logic START

def parse_winner_loser(soup: BeautifulSoup) -> dict:
    """
    Given a UFC Stats fight details soup, return:
        {'winner': {'name': str, 'link': str}, 'loser': {'name': str, 'link': str}}
    """
    result = {'winner': None, 'loser': None}

    try:
        persons = soup.select('div.b-fight-details__person')
        for person in persons:
            status = person.select_one('i.b-fight-details__person-status')
            name_tag = person.select_one('h3.b-fight-details__person-name a')
            if not status or not name_tag:
                continue

            name = name_tag.get_text(strip=True)
            status_text = status.get_text(strip=True).upper()

            if status_text == 'W':
                result['winner'] = name
            elif status_text == 'L':
                result['loser'] = name
    except Exception:
        pass

    return result

def parse_fight_meta_str(s: str) -> dict:
    """
    Parse a line like:
    'Decision - Unanimous Round: 5 Time: 5:00 Time format: 5 Rnd (5-5-5-5-5) Referee: Todd Ronald Anderson'
    
    Returns:
      {
        'method': 'kotko' | 'sub' | 'd_unan' | 'd_split' | 'd_maj' | 'decision' | 'unknown',
        'format': 3|5|None,          # scheduled rounds (int)
        'ref': 'Todd Ronald Anderson'|None,
        'time': 'MM:SS'              # total elapsed fight time up to the stoppage/bell
      }
    """
    txt = s.strip()

    # --- METHOD ---
    method = 'unknown'
    t = txt.lower()
    if 'ko/tko' in t or 'tko' in t or re.search(r'\bko\b', t):
        method = 'kotko'
    elif 'submission' in t:
        method = 'sub'
    elif 'decision' in t:
        if 'unanimous' in t:
            method = 'd_unan'
        elif 'split' in t:
            method = 'd_split'
        elif 'majority' in t:
            method = 'd_maj'
        else:
            method = 'decision'

    # --- FORMAT (scheduled rounds) ---
    # Prefer "Time format: 5 Rnd (...)"  -> 5
    fmt_match = re.search(r'time\s*format\s*:\s*(\d+)\s*rnd', txt, flags=re.I)
    fight_format = int(fmt_match.group(1)) if fmt_match else None

    # --- parse round + intra-round time ---
    rd = None
    mm = 0
    ss = 0

    m_round = re.search(r'round\s*:\s*(\d+)', txt, flags=re.I)
    if m_round:
        rd = int(m_round.group(1))

    m_time = re.search(r'time\s*:\s*(\d{1,2}):(\d{2})', txt, flags=re.I)
    if m_time:
        mm = int(m_time.group(1))
        ss = int(m_time.group(2))

    # per-round minutes (from the parentheses if present, else assume 5)
    per_round_min = 5
    m_paren = re.search(r'R(nd)?\s*\(([\d\-]+)\)', txt, flags=re.I)
    if m_paren:
        # pattern like (5-5-5) or (5-5-5-5-5) -> take the first number
        first_num = re.search(r'(\d+)', m_paren.group(2))
        if first_num:
            per_round_min = int(first_num.group(1))

    # compute elapsed total time
    if rd is not None and m_time:
        elapsed_sec = (rd - 1) * per_round_min * 60 + (mm * 60 + ss)
    else:
        elapsed_sec = None

    def fmt_mmss(total_seconds):
        if total_seconds is None:
            return None
        m, s = divmod(total_seconds, 60)
        return f"{m}:{s:02d}"

    total_time_str = fmt_mmss(elapsed_sec)

    # --- REFEREE ---
    ref = None
    m_ref = re.search(r'referee\s*:\s*([^\n\r]+)$', txt, flags=re.I)
    if m_ref:
        ref = m_ref.group(1).strip()

    return {
        'method': method,
        'format': fight_format,
        'ref': ref,
        'time': total_time_str
    }


def parse_fight_meta(soup: BeautifulSoup) -> dict:
    meta = {
        'method': None,
        'format': None,
        'ref': None,
        'time': None
    }
    def clean_text(node):
        if not node:
            return None
        txt = " ".join(list(node.stripped_strings))
        return txt if txt else None
    try:
        for p in soup.select("p.b-fight-details__text"):
            label_i = p.select_one("i.b-fight-details__label")
            if label_i:
                label_text = (clean_text(label_i) or "").strip().rstrip(":").lower()
                full_text = clean_text(p) or ""
                # value is the text after removing the label
                value = full_text.replace(clean_text(label_i) or "", "", 1).strip()

                if label_text == "method":
                    # Some pages put full string like "Decision - Unanimous" here.
                    meta = parse_fight_meta_str(value)
    except Exception:
        pass
    return meta

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

import re
from datetime import datetime
from bs4 import BeautifulSoup

def _normalize_date(date_text: str) -> str | None:
    """Return MM-DD-YYYY or None."""
    if not date_text:
        return None
    s = date_text.replace("\xa0", " ").strip()
    # Try common UFCStats formats first
    for fmt in ("%b %d, %Y", "%b. %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%m-%d-%Y")
        except ValueError:
            pass
    # Fallback regex (handles 'Oct. 26, 2024' or 'Oct 26, 2024')
    m = re.search(r'([A-Za-z]{3,})\.?\s+(\d{1,2}),\s*(\d{4})', s)
    if m:
        month_map = {
            'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
            'July':7,'August':8,'September':9,'Sept':9,'October':10,'November':11,'December':12,
            'Jan':1,'Feb':2,'Mar':3,'Apr':4,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
        }
        mon = m.group(1).title()
        month_num = month_map.get(mon)
        if month_num:
            return f"{month_num:02d}-{int(m.group(2)):02d}-{m.group(3)}"
    return None

def _normalize_date(date_text: str) -> str | None:
    """Return MM-DD-YYYY or None."""
    if not date_text:
        return None
    s = date_text.replace("\xa0", " ").strip()
    # Try common UFCStats formats first
    for fmt in ("%b %d, %Y", "%b. %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%m-%d-%Y")
        except ValueError:
            pass
    # Fallback regex (handles 'Oct. 26, 2024' or 'Oct 26, 2024')
    m = re.search(r'([A-Za-z]{3,})\.?\s+(\d{1,2}),\s*(\d{4})', s)
    if m:
        month_map = {
            'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
            'July':7,'August':8,'September':9,'Sept':9,'October':10,'November':11,'December':12,
            'Jan':1,'Feb':2,'Mar':3,'Apr':4,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
        }
        mon = m.group(1).title()
        month_num = month_map.get(mon)
        if month_num:
            return f"{month_num:02d}-{int(m.group(2)):02d}-{m.group(3)}"
    return None

def get_fight_links(soup: BeautifulSoup):
    """
    Returns a list of {'link': <fight-details url>, 'date': 'MM-DD-YYYY' or None}.
    Works on UFCStats event pages where each row is clickable.
    """
    fights = []

    # Each clickable row usually has these classes and either data-link or onclick
    rows = soup.select("tbody.b-fight-details__table-body tr.b-fight-details__table-row")

    for row in rows:
        # --- link from data-link or onclick ---
        href = None
        if row.has_attr("data-link"):
            href = row["data-link"].strip()
        if (not href) and row.has_attr("onclick"):
            m = re.search(r"doNav\('([^']+)'", row["onclick"])
            if m:
                href = m.group(1).strip()

        # Final fallback: look for any anchor with fight-details
        if (not href):
            a = row.select_one('a[href*="fight-details"]')
            if a:
                href = a.get("href", "").strip()

        if not href or "fight-details" not in href:
            continue

        # --- date text: scan the row's table-text <p> elements and pick the one that looks like a date ---
        date_str = None
        for p in row.select("p.b-fight-details__table-text"):
            txt = p.get_text(strip=True)
            if re.search(r'[A-Za-z]{3}\.? \d{1,2}, \d{4}', txt) or re.search(r'[A-Za-z]{3,} \d{1,2}, \d{4}', txt):
                date_str = txt
                break

        fights.append({
            "link": href,
            "date": _normalize_date(date_str)
        })

    return fights


### Parsing logic END

### Scrape Requests

def get_single_fight_stats(url: str):
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        winner_loser = parse_winner_loser(soup)
        data2 = parse_fight_meta(soup)
        print(data2)
        return winner_loser
    except Exception as e:
        pass

def get_fighter_data_ufc_stats(fighters_url_ufcstats: str):
    try: 
        response = requests.get(fighters_url_ufcstats, headers=HEADERS)
        # print(response.status_code)
        soup = BeautifulSoup(response.text, "html.parser")
        data = extract_career_stats(soup = soup)
        fight_links = get_fight_links(soup)
        fighter_career_stats = clean_fighter_stats(data)
        count=0
        for i in fight_links:
            if count == 0:
                count+=1
                continue
            data = get_single_fight_stats(i['link'])
            print(f'{i}: {i['date']} {data}')
            count+=1
        # exit()
    except Exception as e: print(e)



def get_fighter_data(fighters_url: str):
    response = requests.get(fighters_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    link=get_ufcstats_link(soup_or_html=soup)
    if link:
        get_fighter_data_ufc_stats(link)
    else:
        print('notwork')