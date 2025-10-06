from datetime import date, datetime
import math
from .utils import fetch_query
from operator import itemgetter

def calc_fight_age_value(d, *, now=None) -> float:
    half_life = 3.25
    floor = 0.3
    
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
        "sub":     0.9,  # Submission
        "d_unan":  0.65,  # Unanimous Decision
        "d_split": 0.35,  # Split Decision
        "d_maj":   0.50,  # Majority Decision
        "unknown": 0.45,  # treat like split decision
        None:      0.45,  # NULL/None -> treat like split decision
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
        "sub":     0.90,  # Submission
        "d_unan":  0.65,  # Unanimous Decision
        "d_split": 0.35,  # Split Decision
        "d_maj":   0.50,  # Majority Decision
        "unknown": 0.45,  # treat like split decision (your choice)
        None:      0.45,  # None -> treat like split decision
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
        S = 100.0    # rating-gap scale (≈ half the 850–1250 spread)
        gamma = 0.5  # max ±50% scaling
        opp_factor = 1.0 + gamma * math.tanh(diff / S)

    # recency already passed in as fight_age_value
    delta = k * m * fight_age_value * opp_factor
    new_elo = fighters_elo + delta if win_loss else fighters_elo - delta
    return new_elo

    
def loop_fights_raw(fighter_id, fights_list):
    elo = 1000
    fights = sorted(fights_list, key=itemgetter('fight_date'))
    for fight in fights:
        if fight.get('method') == None and fight.get('end_time') == None and fight.get('winner_id') == None:
            continue # Fight has not happened yet or results havent been recorded
        fight_age_value = calc_fight_age_value(fight.get('fight_date'))
        winner_id = fight.get('winner_id')
        if fighter_id == winner_id:
            elo = calc_raw_elo(elo, fight.get('method'), fight_age_value, win_loss=True)
        else:
            elo = calc_raw_elo(elo, fight.get('method'), fight_age_value, win_loss=False)
    return elo

def loop_fights_weighted(fighter_id, fights_list, conn):
    get_raw_elo = """
    SELECT raw_elo FROM ufc.advanced_fighter_stats
    WHERE fighter_id = %s
    """
    elo = 1000
    try:
        fighters_elo = float(fetch_query(conn, get_raw_elo, (fighter_id, ))[0].get('raw_elo'))
    except IndexError as e:
        fighters_elo = 1000
        
    fights = sorted(fights_list, key=itemgetter('fight_date'))
    for fight in fights:
        if fight.get('method') == None and fight.get('end_time') == None and fight.get('winner_id') == None:
            continue # Fight has not happened yet or results havent been recorded
        fight_age_value = calc_fight_age_value(fight.get('fight_date'))
        winner_id = fight.get('winner_id')
        f1_id = fight.get('fighter1_id')
        f2_id = fight.get('fighter2_id')
        if f1_id == fighter_id:
            ops_id = f2_id
        else:
            ops_id = f1_id
        try:
            ops_elo = float(fetch_query(conn, get_raw_elo, (ops_id, ))[0].get('raw_elo'))
        except IndexError as e:
            print('op not in advanced ufc stats table, setting ops elo to 1000')
            ops_elo = 1000.0
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