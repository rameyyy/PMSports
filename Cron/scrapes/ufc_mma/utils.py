import re
from bs4 import BeautifulSoup
from datetime import datetime, date
import unicodedata

### BASE LINKS FROM UFC-STATS AND TAPOLOGY ###
ufcstats_fight_details_link = 'http://www.ufcstats.com/fight-details/'
tapology_fighter_profile_link = 'https://www.tapology.com/fightcenter/fighters/'
### END ###


def ufc_weight_class(weight):
    weight = float(weight)
    if weight <= 125:
        return "Flyweight"
    elif weight <= 135:
        return "Bantamweight"
    elif weight <= 145:
        return "Featherweight"
    elif weight <= 155:
        return "Lightweight"
    elif weight <= 170:
        return "Welterweight"
    elif weight <= 185:
        return "Middleweight"
    elif weight <= 205:
        return "Light Heavyweight"
    elif weight <= 265:
        return "Heavyweight"
    return "Super Heavyweight"


def parse_fight_type(fight_str: str):
    """
    Parse a UFC fight description string and return:
      (weight_class, is_title_fight)

    Examples:
      "UFC Bantamweight Title Bout" -> ("Bantamweight", True)
      "Bantamweight Bout" -> ("Bantamweight", False)
      "Catch Weight Bout" -> ("Catch Weight", False)
    """
    fight_str = fight_str.strip()

    # Check if it's a title fight
    is_title = "title" in fight_str.lower()

    # Map substrings to weight classes
    weight_map = {
        "flyweight": "Flyweight",
        "bantamweight": "Bantamweight",
        "featherweight": "Featherweight",
        "lightweight": "Lightweight",
        "welterweight": "Welterweight",
        "middleweight": "Middleweight",
        "light heavyweight": "Light Heavyweight",
        "heavyweight": "Heavyweight",
        "catch weight": "Catch Weight"
    }

    # Find weight class
    weight_class = None
    for key, val in weight_map.items():
        if key in fight_str.lower():
            weight_class = val
            break

    # Default if none matched
    if weight_class is None:
        weight_class = "Unknown"

    return weight_class, is_title


def get_event_title(soup: BeautifulSoup):
    TITLE_HINTS = re.compile(
        r"\b(UFC|Fight Night|Bellator|PFL|Contender|Invicta|One|Cage|LFA)\b", re.I
    )
    h = soup.select_one("h1.text-tap_3, h2.text-tap_3")
    if h and (txt := h.get_text(strip=True)):
        return txt

    for tag in soup.find_all(["h1", "h2"]):
        classes = tag.get("class", [])
        if classes and "text-tap_3" in classes:
            txt = tag.get_text(strip=True)
            if txt:
                return txt

    h = soup.select_one("h1.font-bold.text-center, h2.font-bold.text-center")
    if h and (txt := h.get_text(strip=True)):
        return txt

    h = soup.find(["h1", "h2"], string=lambda s: s and TITLE_HINTS.search(s))
    if h and (txt := h.get_text(strip=True)):
        return txt

    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    return None


def get_event_date_location(soup: BeautifulSoup):
    date, location = None, None
    details = soup.select("ul[data-controller='unordered-list-background'] li")

    for li in details:
        label = li.find("span", class_="font-bold")
        value = li.find("span", class_="text-neutral-700")
        if not label or not value:
            continue

        label_text = label.get_text(strip=True).lower()
        if "date" in label_text:
            date = value.get_text(" ", strip=True)
        elif "location" in label_text:
            location = value.get_text(" ", strip=True)

    return date, location


def parse_event_date(date_str: str):
    cleaned = date_str.replace("ET", "").strip()
    dt = datetime.strptime(cleaned, "%A %m.%d.%Y at %I:%M %p")
    sql_date = dt.strftime("%Y-%m-%d")
    today = date.today()
    is_future = 1 if dt.date() >= today else 0
    return sql_date, is_future


def _normalize_dob(dob_str: str):
    """Convert 'MM-DD-YYYY' -> 'YYYY-MM-DD'. Returns None if invalid/empty."""
    if not dob_str:
        return None
    try:
        dt = datetime.strptime(dob_str, "%m-%d-%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

def _mmss_to_seconds(s):
    """'MM:SS' -> int seconds. Returns None on falsy/invalid."""
    if not s or not isinstance(s, str) or ":" not in s:
        return None
    m, s = s.split(":")
    try:
        return int(m) * 60 + int(s)
    except ValueError:
        return None


def _to_sql_date(date_str: str) -> str:
    """Convert a date string in 'MM-DD-YYYY' format into 'YYYY-MM-DD' for SQL."""
    dt = datetime.strptime(date_str, "%m-%d-%Y")
    return dt.strftime("%Y-%m-%d")


def normalize_name(name: str) -> str:
    # Normalize Unicode characters and strip diacritics
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).lower()


def _normalize_name(name: str) -> list[str]:
    """Normalize a name into cleaned tokens."""
    if not name:
        return []
    # lowercase
    s = name.lower()
    # strip accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # replace hyphens with spaces
    s = s.replace("-", " ")
    # remove punctuation
    s = re.sub(r"[^a-z\s]", " ", s)
    # collapse spaces
    toks = [t for t in s.split() if t]
    return toks


def names_equal(n1: str, n2: str) -> bool:
    """
    Compare two fighter names
    """
    a = set(_normalize_name(n1))
    b = set(_normalize_name(n2))
    return a == b