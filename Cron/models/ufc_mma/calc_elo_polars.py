from datetime import date, datetime
import math
from .utils import fetch_query
from operator import itemgetter
import polars as pl


def lookup_raw_elo(raw_df: pl.DataFrame, fighter_id: str, default: float = 1000.0) -> float:
    # make sure id dtype matches your ops_id type (Utf8 usually)
    row = raw_df.filter(pl.col("id") == fighter_id).select("raw_elo")
    return float(row.item()) if row.height else float(default)

def calc_fight_age_value(d, *, now=None) -> float:
    half_life = 3.5
    floor = 0.4
    
    # choose reference 'now'
    if now is None:
        now = date.today()
    elif isinstance(now, datetime):
        now = now.date()

    days = (now - d).days
    years_ago = days / 365.2425
    f = 0.5 ** (years_ago / half_life)
    return max(f, floor)

def calc_raw_elo(fighters_elo, method, fight_age_value, win_loss = False):
    k = 35 # baseline elo add/sub before multiplying by method tree
    new_elo = 0
    method_tree = {
        "kotko":   1.00,  # KO/TKO
        "sub":     1.0,  # Submission
        "d_unan":  0.85,  # Unanimous Decision
        "d_split": 0.7,  # Split Decision
        "d_maj":   0.8,  # Majority Decision
        "unknown": 0.5,  # treat like split decision
        None:      0.5,  # NULL/None -> treat like split decision
    }
    if win_loss:
        new_elo = fighters_elo + ((k * method_tree.get(method)) * fight_age_value)
    else:
        new_elo = fighters_elo - ((k * method_tree.get(method)) * fight_age_value)
    return new_elo

def calc_weighted_elo(
    fighters_elo,
    method,
    fight_age_value,
    win_loss=False,
    fighters_raw_elo=None,
    ops_raw_elo=None,
):
    k = 35  # baseline KO/TKO bump before multipliers

    # method multipliers (kept exactly like your version)
    method_tree = {
        "kotko":   1.00,  # KO/TKO
        "sub":     0.98,  # Submission
        "d_unan":  0.9,  # Unanimous Decision
        "d_split": 0.75,  # Split Decision
        "d_maj":   0.85,  # Majority Decision
        "unknown": 0.7,  # treat like split decision (your choice)
        None:      0.7,  # None -> treat like split decision
    }

    # safe method lookup
    key = (method or "").strip().lower()
    m = method_tree.get(key, method_tree["d_split"])

    # opponent-strength factor (bounded ~[1-γ, 1+γ])
    # tuned for your raw Elo spread ~850–1250 (half-range ≈ 200)
    if fighters_raw_elo is None or ops_raw_elo is None:
        opp_factor = 1.0
    else:
        diff = ops_raw_elo - fighters_raw_elo    # + if opponent is stronger
        S = 70    # rating-gap scale (≈ half the 850–1250 spread)
        gamma = 0.5  # max ±50% scaling
        opp_factor = 1.0 + gamma * math.tanh(diff / S)

    # recency already passed in as fight_age_value
    delta = k * m * fight_age_value * opp_factor
    new_elo = fighters_elo + delta if win_loss else fighters_elo - delta
    return new_elo

    
def loop_fights_raw(fighter_id, fights_list, date_past):
    elo = 1000
    fights = sorted(fights_list, key=itemgetter('fight_date'))
    for fight in fights:
        if fight.get('method') == None and fight.get('end_time') == None and fight.get('winner_id') == None:
            continue # Fight has not happened yet or results havent been recorded
        fight_age_value = calc_fight_age_value(fight.get('fight_date'), now=date_past)
        winner_id = fight.get('winner_id')
        if fighter_id == winner_id:
            elo = calc_raw_elo(elo, fight.get('method'), fight_age_value, win_loss=True)
        else:
            elo = calc_raw_elo(elo, fight.get('method'), fight_age_value, win_loss=False)
    return elo

def loop_fights_weighted(fighter_id, fights_list, fighters_elo, raw_df, date_past):
    fights = sorted(fights_list, key=itemgetter('fight_date'))
    elo = 1000
    for fight in fights:
        if fight.get('method') == None and fight.get('end_time') == None and fight.get('winner_id') == None:
            continue # Fight has not happened yet or results havent been recorded
        fight_age_value = calc_fight_age_value(fight.get('fight_date'), now=date_past)
        winner_id = fight.get('winner_id')
        f1_id = fight.get('fighter1_id')
        f2_id = fight.get('fighter2_id')
        if f1_id == fighter_id:
            ops_id = f2_id
        else:
            ops_id = f1_id
        ops_elo = lookup_raw_elo(raw_df, ops_id)
        if fighter_id == winner_id:
            elo = calc_weighted_elo(elo, fight.get('method'), fight_age_value, win_loss=True, fighters_raw_elo=fighters_elo, ops_raw_elo=ops_elo)
        else:
            elo = calc_weighted_elo(elo, fight.get('method'), fight_age_value, win_loss=False, fighters_raw_elo=fighters_elo, ops_raw_elo=ops_elo)
    return elo

def get_raw_elo(fighter_id, fights_list):
    raw_elo = loop_fights_raw(fighter_id, fights_list)
    return raw_elo

def get_weighted_elo(fighter_id, fights_list, conn):
    weighted_elo = loop_fights_weighted(fighter_id, fights_list, conn)
    return weighted_elo