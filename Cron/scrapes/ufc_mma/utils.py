import re
from bs4 import BeautifulSoup

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