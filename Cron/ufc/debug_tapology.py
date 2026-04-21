"""
debug_tapology.py — diagnose Tapology scraping issues.
Run from Cron/ufc/: python debug_tapology.py
"""

import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://www.tapology.com"
_scraper = cloudscraper.create_scraper()

for label, url in [
    ("upcoming", f"{BASE_URL}/fightcenter?group=ufc&schedule=upcoming"),
    ("results p1", f"{BASE_URL}/fightcenter?group=ufc&schedule=results&page=1"),
]:
    print(f"\n{'='*60}")
    print(f"[{label}] {url}")
    resp = _scraper.get(url)
    print(f"Status: {resp.status_code}")
    print(f"Response body (first 500 chars):\n{resp.text[:500]}")

    soup = BeautifulSoup(resp.text, "html.parser")

    all_fc_links = [a["href"] for a in soup.find_all("a", href=True) if "fightcenter" in a["href"]]
    print(f"\nAll links containing 'fightcenter' ({len(all_fc_links)} found):")
    for h in all_fc_links[:30]:
        print(f"  {h}")

    event_links = [a["href"] for a in soup.select('a[href^="/fightcenter/events/"]')]
    print(f"\nLinks matching a[href^='/fightcenter/events/'] ({len(event_links)} found):")
    for h in event_links[:10]:
        print(f"  {h}")
