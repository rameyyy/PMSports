import requests
from bs4 import BeautifulSoup, NavigableString
from datetime import datetime
import re
from .utils import parse_fight_type
from difflib import SequenceMatcher
from .sqlpush import fetch_query, create_connection


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
    cleaned['win'] = int(raw['win'])
    cleaned['loss'] = int(raw['loss'])
    cleaned['draw'] = int(raw['draw'])
    return cleaned

### Parsing logic START

def get_fighter_links(soup):
    links = []
    for tag in soup.select("a.b-link.b-fight-details__person-link"):
        href = tag.get("href")
        if href:
            links.append(href.strip())
    return links

# ---------- tiny parsers ----------

def _parse_landed_attempts(s: str):
    # "40 of 152" -> {'landed': 40, 'attempts': 152}
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)\s*$", s or "")
    return {"landed": int(m.group(1)), "attempts": int(m.group(2))} if m else {"landed": None, "attempts": None}


def _parse_int(s: str):
    s = (s or "").strip()
    return None if s in {"---", ""} else int(s)


def _parse_pct(s: str):
    s = (s or "").strip()
    return None if s == "---" else int(s.rstrip("%"))


def _mmss(s: str):
    # normalize "M:SS" or "MM:SS" -> "MM:SS"; allow "---" -> None
    s = (s or "").strip()
    if s in {"---", ""}:
        return None
    m = re.fullmatch(r"(\d+):([0-5]\d)", s)
    if not m:
        return None
    mm = int(m.group(1)); ss = int(m.group(2))
    return f"{mm:02d}:{ss:02d}"

# ---------- factories (avoid aliasing!) ----------

def _landed_attempts():
    return {"landed": None, "attempts": None}


def _sig_breakdown_template():
    # fresh dicts/lists every call
    return {
        "head":     [_landed_attempts(), _landed_attempts()],
        "body":     [_landed_attempts(), _landed_attempts()],
        "leg":      [_landed_attempts(), _landed_attempts()],
        "distance": [_landed_attempts(), _landed_attempts()],
        "clinch":   [_landed_attempts(), _landed_attempts()],
        "ground":   [_landed_attempts(), _landed_attempts()],
    }


def _round_template():
    return {
        "kd": [None, None],
        "sig_str": [_landed_attempts(), _landed_attempts()],
        "sig_str_pct": [None, None],
        "total_str": [_landed_attempts(), _landed_attempts()],
        "td": [_landed_attempts(), _landed_attempts()],
        "td_pct": [None, None],
        "sub_att": [None, None],
        "pass": [None, None],
        "rev": [None, None],
        "ctrl": [None, None],  # "MM:SS"
        "sig_breakdown": _sig_breakdown_template(),
    }

# ---------- main extractor ----------

def extract_fight_stats(soup):
    """Parse Tapology-style 'Totals' + 'Per Round' + 'Significant Strikes' tables."""
    stats = {
        "fighters": [None, None],
        "totals_total": {
            "kd": [None, None],
            "sig_str": [_landed_attempts(), _landed_attempts()],
            "sig_str_pct": [None, None],
            "total_str": [_landed_attempts(), _landed_attempts()],
            "td": [_landed_attempts(), _landed_attempts()],
            "td_pct": [None, None],
            "sub_att": [None, None],
            "pass": [None, None],     # Tapology sometimes omits; we detect below
            "rev": [None, None],
            "ctrl": [None, None],
            "sig_breakdown": _sig_breakdown_template(),
        },
        "rounds": {f"round{i}": _round_template() for i in range(1, 6)},
    }

    stream = [p.get_text(strip=True) for p in soup.find_all("p", class_="b-fight-details__table-text")]

    i = 0
    def nxt():
        nonlocal i
        if i >= len(stream): return None
        v = stream[i]; i += 1; return v
    def peek(k=0):
        j = i + k
        return stream[j] if j < len(stream) else None

    # ---- Phase A: TOTALS (overall) ----
    stats["fighters"][0] = nxt()
    stats["fighters"][1] = nxt()

    def fill_totals_into(target):
        # order exactly matches your stream
        target["kd"][0] = _parse_int(nxt())
        target["kd"][1] = _parse_int(nxt())

        target["sig_str"][0] = _parse_landed_attempts(nxt())
        target["sig_str"][1] = _parse_landed_attempts(nxt())

        target["sig_str_pct"][0] = _parse_pct(nxt())
        target["sig_str_pct"][1] = _parse_pct(nxt())

        target["total_str"][0] = _parse_landed_attempts(nxt())
        target["total_str"][1] = _parse_landed_attempts(nxt())

        target["td"][0] = _parse_landed_attempts(nxt())
        target["td"][1] = _parse_landed_attempts(nxt())

        target["td_pct"][0] = _parse_pct(nxt())
        target["td_pct"][1] = _parse_pct(nxt())

        target["sub_att"][0] = _parse_int(nxt())
        target["sub_att"][1] = _parse_int(nxt())

        # Optional "pass" column: look ahead safely
        def _looks_int(x): 
            return x is not None and re.fullmatch(r"\d+|---", x) is not None
        def _looks_time(x):
            return x is not None and re.fullmatch(r"\d+:[0-5]\d|---", x) is not None

        # If next two are ints and the third is not a time, treat them as PASS values.
        if _looks_int(peek(0)) and _looks_int(peek(1)) and not _looks_time(peek(2)):
            target["pass"][0] = _parse_int(nxt())
            target["pass"][1] = _parse_int(nxt())

        target["rev"][0] = _parse_int(nxt())
        target["rev"][1] = _parse_int(nxt())

        target["ctrl"][0] = _mmss(nxt())
        target["ctrl"][1] = _mmss(nxt())

    fill_totals_into(stats["totals_total"])

    # ---- Phase B: PER-ROUND totals ----
    round_idx = 1
    while round_idx <= 5 and i < len(stream):
        if peek() != stats["fighters"][0]:
            break
        # consume names for each round block
        name_a = nxt(); name_b = nxt()

        # If the next token is "X of Y", we've reached the significant-strikes section; rewind names and stop.
        if " of " in (peek() or ""):
            i -= 2
            break

        fill_totals_into(stats["rounds"][f"round{round_idx}"])
        round_idx += 1

    # ---- Phase C: SIGNIFICANT STRIKES (totals) ----
    # names again
    if nxt() != stats["fighters"][0]: 
        return stats
    if nxt() != stats["fighters"][1]: 
        return stats

    stats["totals_total"]["sig_str"][0] = _parse_landed_attempts(nxt())
    stats["totals_total"]["sig_str"][1] = _parse_landed_attempts(nxt())
    stats["totals_total"]["sig_str_pct"][0] = _parse_pct(nxt())
    stats["totals_total"]["sig_str_pct"][1] = _parse_pct(nxt())

    for bucket in ("head","body","leg","distance","clinch","ground"):
        stats["totals_total"]["sig_breakdown"][bucket][0] = _parse_landed_attempts(nxt())
        stats["totals_total"]["sig_breakdown"][bucket][1] = _parse_landed_attempts(nxt())

    # ---- Phase D: SIGNIFICANT STRIKES (per round) ----
    r = 1
    while r <= 5 and i < len(stream):
        name_a = nxt(); name_b = nxt()
        if name_a != stats["fighters"][0] or name_b != stats["fighters"][1]:
            break

        node = stats["rounds"][f"round{r}"]
        node["sig_str"][0] = _parse_landed_attempts(nxt())
        node["sig_str"][1] = _parse_landed_attempts(nxt())
        node["sig_str_pct"][0] = _parse_pct(nxt())
        node["sig_str_pct"][1] = _parse_pct(nxt())

        for bucket in ("head","body","leg","distance","clinch","ground"):
            node["sig_breakdown"][bucket][0] = _parse_landed_attempts(nxt())
            node["sig_breakdown"][bucket][1] = _parse_landed_attempts(nxt())

        r += 1

    return stats


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
        'type': None,
        'ref': ref,
        'time': total_time_str,
        'weight_class': None
    }


def parse_fight_meta(soup: BeautifulSoup) -> dict:
    meta = {
        'method': None,
        'format': None,
        'type': None,
        'ref': None,
        'time': None,
        'weight_class': None
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
        for p in soup.select('i.b-fight-details__fight-title'):
            fight_name = p.text.strip()
            weightclass, title_bool = parse_fight_type(fight_name)
            if title_bool:
                meta['type'] = 'title'
            elif meta['format'] == 5:
                meta['type'] = 'main'
            else:
                meta['type'] = None
            if weightclass == 'Unknown':
                raise f"Weight class should never be unknown. soup text == {fight_name}"
            else:
                meta['weight_class'] = weightclass
        if meta['weight_class'] == None:
            return -1
    except Exception:
        pass
    return meta

def get_fname_from_fighter_details(soup):
    for p in soup.select('span.b-content__title-highlight'):
        raw = p.text.strip()
        return raw

def extract_career_stats(soup):
    stats = {}
    
    # Find the "Career statistics" box
    stat_items = soup.select("ul.b-list__box-list li.b-list__box-list-item")

    for p in soup.select('span.b-content__title-record'):
        raw = p.text.strip()
        WLD = raw.split(' ')[1]
        win, loss, draw = WLD.split('-')
        stats['win'] = win
        stats['loss'] = loss
        stats['draw'] = draw
    
    for item in stat_items:
        label_tag = item.find("i", class_="b-list__box-item-title")
        if label_tag:
            label = label_tag.get_text(strip=True)
            value = label_tag.next_sibling.strip() if label_tag.next_sibling else None
            
            if label and value:
                stats[label] = value
    
    return stats


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


def least_similar(given: str, candidates: list[str]) -> str:
    scores = [(c, SequenceMatcher(None, given, c).ratio()) for c in candidates]
    return min(scores, key=lambda x: x[1])[0]

### Parsing logic END


### Scrape Requests
    
def get_single_fight_stats(url: str, date, og_link, fname):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    resp_code = response.status_code
    
    # Collect all the different dicts
    fighter_links = get_fighter_links(soup)
    ops_url = ''
    for link in fighter_links:
        if link != og_link:
            ops_url = link
    ops_resp = requests.get(ops_url, headers=HEADERS)
    ops_soup = BeautifulSoup(ops_resp.text, 'html.parser')

    ops_career_stats = extract_career_stats(ops_soup)
    ops_career_stats_cleaned = clean_fighter_stats(ops_career_stats)
    winner_loser = parse_winner_loser(soup)     # winner/loser names
    meta        = parse_fight_meta(soup)        # method, ref, time, format
    if meta == -1:
        print(f'err at: {url}')
    fight_stats = extract_fight_stats(soup)     # totals + rounds stats

    fighters_arr = fight_stats['fighters']
    op_name = least_similar(fname, fighters_arr)
    
    ops_career_stats_cleaned['fighter_name'] = op_name
    ops_career_stats_cleaned['fighter_id'] = ops_url.rstrip("/").split("/")[-1]

    # Merge into one dictionary
    fight_data = {
        "url": url,
        "fight_id": url.rstrip("/").split("/")[-1],
        "date": date,
        "winner_loser": winner_loser,
        "meta": meta,
        f'ops_careerstats': ops_career_stats_cleaned,
        "stats": fight_stats
    }
    return fight_data, resp_code


def get_fighter_data_ufc_stats(fighters_url_ufcstats: str, fname, conn):
    response = requests.get(fighters_url_ufcstats, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    data = extract_career_stats(soup = soup)
    fight_links = get_fight_links(soup)
    fighter_career_stats = clean_fighter_stats(data)
    fighter_career_stats['fighter_name'] = fname
    fid = fighters_url_ufcstats.rstrip("/").split("/")[-1]
    fighter_career_stats['fighter_id'] = fid
    fights_arr = []
    upcoming = []
    cmd = """
    SELECT *
    FROM ufc.fights
    WHERE fight_id = %s
    """
    skipped = 0
    for i in fight_links:
        try:
            fightid = i['link'].rstrip("/").split("/")[-1]
            rows = fetch_query(conn, cmd, (fightid,))
            if rows:
                if rows[0].get('winner_id') != None:
                    skipped+=1
                    resp_code = 200
                    continue
            data, resp_code = get_single_fight_stats(i['link'], i['date'], fighters_url_ufcstats, fname)
            fights_arr.append(data)
        except ValueError as e:
            # Only catch the int conversion error you mentioned
            if "invalid literal for int()" in str(e):
                response_ = requests.get(i['link'], headers=HEADERS)
                soup_ = BeautifulSoup(response_.text, "html.parser")
                links = get_fighter_links(soup_)
                if links:
                    for link in links:
                        if fid not in link:
                            fight_idt = i['link'].rstrip("/").split("/")[-1]
                            opp_fid = link.rstrip("/").split("/")[-1]
                            response_1 = requests.get(link, headers=HEADERS)
                            soup_1 = BeautifulSoup(response_1.text, "html.parser")
                            f2_careerstats = extract_career_stats(soup_1)
                            f2_careerstats = clean_fighter_stats(f2_careerstats)
                            f2_careerstats['fighter_id'] = opp_fid
                            f2_careerstats['fighter_name'] = get_fname_from_fighter_details(soup_1)
                            upcoming.append({'fight_id': fight_idt, 'fighter1_name': fname, 'fighter1_id': fid, 'fighter2_careerstats': f2_careerstats})
                            break
                continue
            else:
                # If it's some other ValueError, re-raise so you don’t hide real bugs
                raise
    return fighter_career_stats, fights_arr, upcoming, resp_code


def get_fighter_data(fighters_url: str, fname):
    response = requests.get(fighters_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    link=get_ufcstats_link(soup_or_html=soup)
    if link:
        conn = create_connection()
        fighter_career_stats, fights_arr, upcoming, resp_code = get_fighter_data_ufc_stats(link, fname, conn)
        return fighter_career_stats, fights_arr, upcoming, resp_code
    return None, None, None, 200