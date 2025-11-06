#!/usr/bin/env python3
"""
Check for future data leakage in sample2.parquet
"""
import polars as pl
from datetime import timedelta

# Load sample2.parquet
df = pl.read_parquet("sample2.parquet")

print("=" * 80)
print("CHECKING FOR FUTURE DATA LEAKAGE")
print("=" * 80)

leakage_found = []

# Check each game
for i, row in enumerate(df.iter_rows(named=True)):
    game_id = row['game_id']
    game_date = row['date']
    team_1 = row['team_1']
    team_2 = row['team_2']

    # Check team_1 match history
    team_1_hist = row['team_1_match_hist']
    if team_1_hist:
        for j, hist_game in enumerate(team_1_hist):
            hist_date = hist_game.get('date')
            if hist_date and hist_date > game_date:
                leakage_found.append({
                    'issue': 'Future game in team_1 history',
                    'game_id': game_id,
                    'game_date': game_date,
                    'team': team_1,
                    'hist_game_date': hist_date,
                    'hist_opponent': hist_game.get('team_2'),
                })

    # Check team_2 match history
    team_2_hist = row['team_2_match_hist']
    if team_2_hist:
        for j, hist_game in enumerate(team_2_hist):
            hist_date = hist_game.get('date')
            if hist_date and hist_date > game_date:
                leakage_found.append({
                    'issue': 'Future game in team_2 history',
                    'game_id': game_id,
                    'game_date': game_date,
                    'team': team_2,
                    'hist_game_date': hist_date,
                    'hist_opponent': hist_game.get('team_1'),
                })

if leakage_found:
    print(f"\nFOUND {len(leakage_found)} FUTURE DATA LEAKAGE ISSUES!\n")
    for issue in leakage_found[:10]:  # Show first 10
        print(f"Issue: {issue['issue']}")
        print(f"  Game: {issue['game_id']} on {issue['game_date']}")
        print(f"  Team: {issue['team']}")
        print(f"  Future game date: {issue['hist_game_date']} vs {issue['hist_opponent']}")
        print()
else:
    print("\nNo future games found in match histories - GOOD!")

# Check leaderboard data
print("\n" + "=" * 80)
print("CHECKING LEADERBOARD DATA TIMING")
print("=" * 80)

leaderboard_leakage = []

for i, row in enumerate(df.iter_rows(named=True)):
    game_date = row['date']
    team_1_lb = row['team_1_leaderboard']
    team_2_lb = row['team_2_leaderboard']

    # Team 1 leaderboard should be from day before
    if team_1_lb and isinstance(team_1_lb, dict):
        lb_date = team_1_lb.get('date')
        if lb_date:
            day_before = game_date - timedelta(days=1)
            if lb_date != day_before:
                leaderboard_leakage.append({
                    'game_id': row['game_id'],
                    'game_date': game_date,
                    'team': row['team_1'],
                    'lb_date': lb_date,
                    'expected_date': day_before,
                    'issue': 'Team 1 leaderboard not from day before'
                })

    # Team 2 leaderboard should be from day before
    if team_2_lb and isinstance(team_2_lb, dict):
        lb_date = team_2_lb.get('date')
        if lb_date:
            day_before = game_date - timedelta(days=1)
            if lb_date != day_before:
                leaderboard_leakage.append({
                    'game_id': row['game_id'],
                    'game_date': game_date,
                    'team': row['team_2'],
                    'lb_date': lb_date,
                    'expected_date': day_before,
                    'issue': 'Team 2 leaderboard not from day before'
                })

if leaderboard_leakage:
    print(f"\nFound {len(leaderboard_leakage)} leaderboard timing issues\n")
    for issue in leaderboard_leakage[:10]:
        print(f"Game: {issue['game_id']} on {issue['game_date']}")
        print(f"  Team: {issue['team']}")
        print(f"  LB Date: {issue['lb_date']}, Expected: {issue['expected_date']}")
        print()
else:
    print("Leaderboard data timing looks correct - GOOD!")

# Check historical game leaderboard data
print("\n" + "=" * 80)
print("CHECKING HISTORICAL GAME LEADERBOARD DATA")
print("=" * 80)

hist_lb_leakage = []

for i, row in enumerate(df.iter_rows(named=True)):
    game_date = row['date']

    # Check team_1 match history leaderboards
    team_1_hist = row['team_1_match_hist']
    if team_1_hist:
        for hist_game in team_1_hist:
            hist_game_date = hist_game.get('date')

            # Check team_1_leaderboard in historical game
            hist_t1_lb = hist_game.get('team_1_leaderboard')
            if hist_t1_lb and isinstance(hist_t1_lb, dict):
                hist_lb_date = hist_t1_lb.get('date')
                if hist_lb_date:
                    expected = hist_game_date - timedelta(days=1)
                    if hist_lb_date != expected:
                        hist_lb_leakage.append({
                            'current_game': row['game_id'],
                            'current_game_date': game_date,
                            'hist_game_date': hist_game_date,
                            'hist_lb_date': hist_lb_date,
                            'expected': expected,
                            'issue': f'Hist game {hist_game_date} has wrong LB date'
                        })

if hist_lb_leakage:
    print(f"\nFound {len(hist_lb_leakage)} historical leaderboard issues\n")
    for issue in hist_lb_leakage[:5]:
        print(f"Current game: {issue['current_game']} ({issue['current_game_date']})")
        print(f"  Historical game: {issue['hist_game_date']}")
        print(f"  LB Date: {issue['hist_lb_date']}, Expected: {issue['expected']}")
        print()
else:
    print("Historical game leaderboard timing looks correct - GOOD!")

# Check scores
print("\n" + "=" * 80)
print("CHECKING FOR SUSPICIOUS DATA PATTERNS")
print("=" * 80)

suspicious = []

for i, row in enumerate(df.iter_rows(named=True)):
    game_id = row['game_id']
    team_1_score = row['team_1_score']
    team_2_score = row['team_2_score']
    team_1_winloss = row['team_1_winloss']

    # Check if winloss label matches scores
    if team_1_score is not None and team_2_score is not None:
        if team_1_score > team_2_score and team_1_winloss != 1:
            suspicious.append(f"Game {game_id}: Team 1 won but winloss={team_1_winloss}")
        elif team_1_score < team_2_score and team_1_winloss != 0:
            suspicious.append(f"Game {game_id}: Team 1 lost but winloss={team_1_winloss}")
        elif team_1_score == team_2_score:
            suspicious.append(f"Game {game_id}: Tied game (rare in basketball)")

if suspicious:
    print(f"\nFound {len(suspicious)} suspicious patterns:")
    for s in suspicious[:5]:
        print(f"  - {s}")
else:
    print("Score labels look consistent - GOOD!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total games checked: {len(df)}")
print(f"Future data leakage in match histories: {len(leakage_found)}")
print(f"Leaderboard timing issues: {len(leaderboard_leakage)}")
print(f"Historical LB timing issues: {len(hist_lb_leakage)}")
print(f"Suspicious patterns: {len(suspicious)}")

if not any([leakage_found, leaderboard_leakage, hist_lb_leakage, suspicious]):
    print("\nNo data leakage detected! Dataset looks clean.")
