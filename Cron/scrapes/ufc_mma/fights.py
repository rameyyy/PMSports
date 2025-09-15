from __future__ import annotations
import requests
from bs4 import BeautifulSoup, NavigableString
from typing import Any, Dict, List, Optional, Tuple
import re

BASE_URL = "https://www.tapology.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

### Parsing logic START

### Parsing logic END

def get_fighter_data(event_url: str):
    response = requests.get(event_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    # pass soup into parser, return then write data