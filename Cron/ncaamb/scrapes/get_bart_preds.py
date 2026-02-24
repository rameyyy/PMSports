"""
Compare Barttorvik win probabilities against our moneyline model predictions.

Scrapes Barttorvik's super_sked for 2026, builds game_ids matching our schema
(YYYYMMDD_team1_team2 alphabetically sorted), then joins against the moneyline
table to see what % of predictions agree on the winner.

NOTE: Indices 12/18 (t1wp/t2wp) get overwritten to 0/1 after a game completes.
Indices 52/53 (t1fun/t2fun) retain the pre-game win probabilities permanently.
"""

import pandas as pd
from . import sqlconn
from .barttorvik_schedule import scrape_barttorvik_schedule


def build_bart_preds(season='2026'):
    """
    Scrape Barttorvik and build a DataFrame of game_id, bart_team_1_wp, bart_team_2_wp.

    Uses indices 52/53 (t1fun/t2fun) which hold the persistent pre-game win
    probabilities even after the game is completed.

    Barttorvik's team1/team2 are NOT alphabetically sorted, but our game_id uses
    alphabetical order. So we need to map probs to the correct side.
    """
    df = scrape_barttorvik_schedule(season)
    if df is None or len(df) == 0:
        print("No data from Barttorvik")
        return None

    rows = []
    for _, row in df.iterrows():
        try:
            bart_t1 = str(row['team1']).strip()
            bart_t2 = str(row['team2']).strip()
            # Use t1fun/t2fun (idx 52/53) — persistent pre-game win probs
            t1wp = row['t1fun']
            t2wp = row['t2fun']

            # Skip if win probs are missing
            if pd.isna(t1wp) or pd.isna(t2wp):
                continue

            t1wp = float(t1wp)
            t2wp = float(t2wp)

            # Skip if both are 0 (bad data)
            if t1wp == 0 and t2wp == 0:
                continue

            # Build game_id with alphabetical team order
            date_obj = pd.to_datetime(row['date'], format='%m/%d/%y')
            date_str = date_obj.strftime('%Y%m%d')
            teams_sorted = sorted([bart_t1, bart_t2])
            game_id = f"{date_str}_{teams_sorted[0]}_{teams_sorted[1]}"

            # Map bart probs to our team_1 / team_2 (alphabetical)
            if bart_t1 == teams_sorted[0]:
                # bart's team1 IS our team_1
                bart_team_1_wp = t1wp
                bart_team_2_wp = t2wp
            else:
                # bart's team1 is our team_2, so flip
                bart_team_1_wp = t2wp
                bart_team_2_wp = t1wp

            rows.append({
                'game_id': game_id,
                'team_1': teams_sorted[0],
                'team_2': teams_sorted[1],
                'bart_team_1_wp': bart_team_1_wp,
                'bart_team_2_wp': bart_team_2_wp,
            })
        except Exception as e:
            continue

    bart_df = pd.DataFrame(rows)
    # Drop duplicate game_ids (each game appears twice in super_sked)
    bart_df = bart_df.drop_duplicates(subset='game_id', keep='first')
    print(f"Built {len(bart_df)} Barttorvik predictions")
    return bart_df


def compare_with_moneyline(bart_df):
    """
    Join Barttorvik preds against our moneyline table and compare winner picks.
    Only compares games that exist in the moneyline table.
    """
    conn = sqlconn.create_connection()
    if not conn:
        print("Could not connect to database")
        return None

    ml_rows = sqlconn.fetch(conn, """
        SELECT game_id, team_1, team_2,
               ensemble_prob_team_1, ensemble_prob_team_2,
               team_predicted_to_win, winning_team
        FROM moneyline
        WHERE season = 2026
    """)
    conn.close()

    if not ml_rows:
        print("No moneyline rows for 2026")
        return None

    ml_df = pd.DataFrame(ml_rows)
    ml_df = ml_df.drop_duplicates(subset='game_id', keep='first')
    print(f"Loaded {len(ml_df)} moneyline games")

    # Filter bart to only games in moneyline
    bart_df = bart_df[bart_df['game_id'].isin(ml_df['game_id'])].copy()
    print(f"Filtered Barttorvik to {len(bart_df)} matching games")

    # Check for moneyline games missing from bart
    missing = ml_df[~ml_df['game_id'].isin(bart_df['game_id'])]
    if len(missing) > 0:
        print(f"\n{len(missing)} moneyline games NOT found in Barttorvik:")
        for _, row in missing.iterrows():
            print(f"  {row['game_id']}")
        print()

    # Join
    merged = bart_df.merge(ml_df, on='game_id', suffixes=('_bart', '_ml'))

    # Determine who each model picks to win
    merged['bart_pick'] = merged.apply(
        lambda r: r['team_1_bart'] if r['bart_team_1_wp'] > r['bart_team_2_wp'] else r['team_2_bart'],
        axis=1
    )
    merged['our_pick'] = merged['team_predicted_to_win']

    # Compare
    merged['agree'] = merged['bart_pick'] == merged['our_pick']
    agree_pct = merged['agree'].mean() * 100

    print(f"Agreement: {merged['agree'].sum()} / {len(merged)} = {agree_pct:.1f}%\n")

    # Show disagreements
    disagree = merged[~merged['agree']]
    if len(disagree) > 0:
        print(f"Disagreements ({len(disagree)}):")
        for _, row in disagree.head(20).iterrows():
            print(f"  {row['game_id']}")
            print(f"    Bart: {row['bart_pick']} ({row['bart_team_1_wp']:.1%} vs {row['bart_team_2_wp']:.1%})")
            print(f"    Ours: {row['our_pick']} ({float(row['ensemble_prob_team_1']):.1%} vs {float(row['ensemble_prob_team_2']):.1%})")

    # --- ACCURACY: compare against actual winners ---
    completed = merged[merged['winning_team'].notna() & (merged['winning_team'] != '')].copy()
    if len(completed) > 0:
        completed['bart_correct'] = completed['bart_pick'] == completed['winning_team']
        completed['our_correct'] = completed['our_pick'] == completed['winning_team']

        bart_acc = completed['bart_correct'].mean() * 100
        our_acc = completed['our_correct'].mean() * 100

        print(f"\n{'=' * 60}")
        print(f"ACCURACY (completed games: {len(completed)})")
        print(f"{'=' * 60}")
        print(f"Barttorvik: {completed['bart_correct'].sum()} / {len(completed)} = {bart_acc:.1f}%")
        print(f"Our model:  {completed['our_correct'].sum()} / {len(completed)} = {our_acc:.1f}%")

    return merged


if __name__ == "__main__":
    bart_df = build_bart_preds('2026')
    if bart_df is not None:
        compare_with_moneyline(bart_df)
