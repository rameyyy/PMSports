from models.ufc_mma import elo, calc_elo_polars
from scrapes.ufc_mma import get_event_data, fetch_query, create_connection, namematch
import polars as pl
event = 'https://www.tapology.com/fightcenter/events/129311-ufc-320'
# event_data = get_event_data(event, True)
import json
# with open("events.json", "w", encoding="utf-8") as f:
#     json.dump(event_data, f, indent=2, ensure_ascii=False)
import json

with open("events.json", "r", encoding="utf-8") as f:
    data = json.load(f)   # dict or list

fights_list = data.get('fights')
conn = create_connection()
sql = """
SELECT *
FROM ufc.fighters
WHERE name LIKE %s
"""
sql2 = """
SELECT * FROM ufc.advanced_fighter_stats
WHERE fighter_id = %s
"""
count=0
query = """
SELECT * FROM ufc.fights
"""
query2 = """
SELECT * FROM ufc.fighters
"""
import time
start_time = time.time()
conn = create_connection()
fights = fetch_query(conn, query)
fighters = fetch_query(conn, query2)
fights_df = pl.DataFrame(fights)
fighters_df = pl.DataFrame(fighters)


# 0) Ensure compatible dtypes for joins/filters (IDs as Utf8)
fights_df = fights_df.with_columns(
    pl.col("fighter1_id").cast(pl.Utf8),
    pl.col("fighter2_id").cast(pl.Utf8),
)
fighters_df = fighters_df.with_columns(
    pl.col("fighter_id").cast(pl.Utf8)
)

# 1) Count appearances in fights (as fighter1 OR fighter2)
fight_counts = (
    pl.concat([
        fights_df.select(pl.col("fighter1_id").alias("fighter_id")),
        fights_df.select(pl.col("fighter2_id").alias("fighter_id")),
    ])
    .group_by("fighter_id")
    .len()
    .rename({"len": "fight_count"})
)

# 2) Keep only fighters with >= 10 fights
fighters_df = (
    fighters_df
    .join(fight_counts, on="fighter_id", how="left")
    .with_columns(pl.col("fight_count").fill_null(0))
    .filter(pl.col("fight_count") >= 10)
)

# 3) Build a Python list of eligible fighter IDs (avoid Series to silence deprecation)
eligible_ids = fighters_df.get_column("fighter_id").unique().to_list()  # <-- list, not Series

# 4) Filter fights: both participants must be in eligible list
fights_df = fights_df.filter(
    pl.col("fighter1_id").is_in(eligible_ids)
    & pl.col("fighter2_id").is_in(eligible_ids)
)


# ensure it's a Date (optional if already Date/Datetime)
f = fights_df.with_columns(pl.col("fight_date").cast(pl.Date))

date_counts = (
    f.group_by("fight_date")
     .len()
     .rename({"len": "n"})
     .sort("fight_date")
)

import polars as pl

# init once
raw_elo = pl.DataFrame(schema={"id": pl.Utf8, "raw_elo": pl.Float64, "weighted_elo": pl.Float64})
seen = set()

print('getting raw_elo')
### GET RAW ELO ###
# def _append_raw(fid: str, elo_val: float):
#     global raw_elo, seen
#     if fid in seen:
#         return
#     row = pl.DataFrame(
#         {"id": [fid], "raw_elo": [float(elo_val)], "weighted_elo": [None]}
#     ).with_columns(pl.col("weighted_elo").cast(pl.Float64))
#     raw_elo = raw_elo.vstack(row)
#     seen.add(fid)
    
# for fight_date, n in date_counts.iter_rows():
#     fights_before = fights_df.filter(pl.col("fight_date") < fight_date)
#     # unique fighter ids in this subset (list, not Series)
#     ids_in_subset = (
#         pl.concat([
#             fights_before.select(pl.col("fighter1_id").alias("id")),
#             fights_before.select(pl.col("fighter2_id").alias("id")),
#         ])
#         .select(pl.col("id").drop_nulls().unique())
#         .to_series()
#         .to_list()
#     )

#     # iterate unique IDs, not rows
#     for fid in ids_in_subset:
#         if fid in seen:
#             continue
#         fighter_fights = (
#             fights_before
#             .filter((pl.col("fighter1_id") == fid) | (pl.col("fighter2_id") == fid))
#             .sort("fight_date")
#             .to_dicts()
#         )
#         elo = calc_elo_polars.loop_fights_raw(fid, fighter_fights, fight_date)
#         _append_raw(fid, elo)

#     # write and reset (NOTE the reassignment!)
#     raw_elo.write_csv(f"temp/{fight_date}.csv")
#     raw_elo = raw_elo.clear()   # <— reassign; clear returns a new empty DF
#     seen.clear()
### GET RAW ELO END ###

print('getting weighted_elo')
### GET WEIGHTED ELO ###
def _append_raw(fid: str, elo_val: float, weighted_val=None):
    global raw_elo, seen
    if fid in seen:
        return
    row = pl.DataFrame(
        {"id": [fid], "raw_elo": [float(elo_val)], "weighted_elo": [weighted_val]}
    ).with_columns(pl.col("weighted_elo").cast(pl.Float64))
    raw_elo = raw_elo.vstack(row)
    seen.add(fid)

all_rows = []
from pathlib import Path
base = Path('temp')
for fight_date, _ in date_counts.iter_rows():
    p = base / f"{fight_date}.csv"
    if not p.exists(): continue
    df = pl.read_csv(p, dtypes={"id": pl.Utf8, "raw_elo": pl.Float64, "weighted_elo": pl.Float64})
    df = df.unique(subset=["id"], keep="first")
    raw_elo_dict = df.rows()
    fights_before = fights_df.filter(pl.col("fight_date") < fight_date)
    for row in raw_elo_dict:
        fid = row[0]
        fighter_raw_elo = row[1]
        if fid in seen:
            continue
        fighter_fights = (
            fights_before
            .filter((pl.col("fighter1_id") == fid) | (pl.col("fighter2_id") == fid))
            .sort("fight_date")
            .to_dicts()
        )
        elo_weighted = calc_elo_polars.loop_fights_weighted(fid, fighter_fights, fighter_raw_elo, df, fight_date)
        _append_raw(fid, fighter_raw_elo, elo_weighted)

    # # write and reset (NOTE the reassignment!)
    raw_elo.write_csv(f"temp/{fight_date}.csv")
    raw_elo = raw_elo.clear()   # <— reassign; clear returns a new empty DF
    seen.clear()
### GET WEIGHTED ELO END ###

print('getting accuracy')
### TEST ACCURACY ###
count_total_total = 0
correct_raw_total = 0
correct_weighted_total = 0
from pathlib import Path
base = Path('temp')
for fight_date, n in date_counts.iter_rows():
    fights_today = fights_df.filter(pl.col("fight_date") == fight_date)
    p = base / f"{fight_date}.csv"
    if not p.exists(): continue
    df = pl.read_csv(p, dtypes={"id": pl.Utf8, "raw_elo": pl.Float64, "weighted_elo": pl.Float64})
    # fast lookup maps
    id_to_raw = dict(zip(df.get_column("id").to_list(),
                         df.get_column("raw_elo").to_list()))
    id_to_weighted = dict(zip(df.get_column("id").to_list(),
                              df.get_column("weighted_elo").to_list()))
    
    count_total = 0
    correct_raw = 0
    correct_weighted = 0
    
    for row in fights_today.rows():
        # row layout you showed:
        # (fight_id, event_id, fighter1_id, fighter2_id, winner_id, loser_id, ... fight_date, ... method ...)
        f1_id = row[2]
        f2_id = row[3]
        winner_id = row[4]
        if winner_id == None:
            # print('dont know winner yet')
            continue

        # look up ratings (defaults if missing)
        f1_raw = id_to_raw.get(f1_id, None)
        f2_raw = id_to_raw.get(f2_id, None)
        
        if f1_raw == None or f2_raw == None:
            # print(f1_id, f2_id, 'one is missing')
            continue

        f1_w = id_to_weighted.get(f1_id, None)
        f2_w = id_to_weighted.get(f2_id, None)

        # RAW prediction
        pred_raw = f1_id if f1_raw >= f2_raw else f2_id
        if pred_raw == winner_id:
            correct_raw += 1

        
        # WEIGHTED prediction (treat None as baseline 1000.0 so it still predicts)
        w1 = f1_w
        w2 = f2_w
        pred_w = f1_id if w1 >= w2 else f2_id
        if pred_w == winner_id:
            correct_weighted += 1

        count_total += 1

    if count_total > 0:
        pct_raw = 100.0 * correct_raw / count_total
        pct_w = 100.0 * correct_weighted / count_total
        # print(f"{fight_date}: raw={pct_raw:.2f}%  weighted={pct_w:.2f}%  (n={count_total})")
    count_total_total += count_total
    correct_raw_total += correct_raw
    correct_weighted_total += correct_weighted
pct_raw_total = 100.0 * correct_raw_total / count_total_total
pct_w_total = 100.0 * correct_weighted_total / count_total_total
print(f"TOTAL: raw={pct_raw_total:.2f}%  weighted={pct_w_total:.2f}%  (n={count_total_total})")
### TEST ACCURACY END ###

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.2f} seconds")
# exit()
# for fight in fights_list:
#     f1 = fight.get('fighter1')
#     f1_name = f1.get('fighter_name')
#     f2 = fight.get('fighter2')
#     f2_name = f2.get('fighter_name')
#     params1 = (f"%{f1_name}%",)
#     params2 = (f"%{f2_name}%",)
#     f1_data = fetch_query(conn, sql, params1)
#     f2_data = fetch_query(conn, sql, params2)
#     if not f1_data or not f2_data:
#         continue
#     f1_id = f1_data[0].get('fighter_id')
#     f2_id = f2_data[0].get('fighter_id')
#     f1_elo = fetch_query(conn, sql2, (f1_id, ))[0]
#     f2_elo = fetch_query(conn, sql2, (f2_id, ))[0]
#     f1_raw, f1_weighted = float(f1_elo.get('raw_elo')), float(f1_elo.get('weighted_elo'))
#     f2_raw, f2_weighted = float(f2_elo.get('raw_elo')), float(f2_elo.get('weighted_elo'))
#     if f1_weighted > f2_weighted:
#         winner = f1_name
#     else:
#         winner = f2_name
#     print(f'{f1_name} vs {f2_name}')
#     print(f'Predicted winner: {winner}')
#     print(f'WEIGHTED: {f1_name}: {f1_weighted:.2f} | {f2_name}: {f2_weighted:.2f}')
#     print(f'RAW: {f1_name}: {f1_raw:.2f} | {f2_name}: {f2_raw:.2f}')
#     print()
#     count+=1
# print(count)
    
# print(event_data)
# elo.raw_elo()