# How team_1 and team_2 are Assigned

## Step 1: Schedule Order (ou_main.py)
The schedule has columns: `team1`, `team2`, `date`
- `team_1` = `row['team1']` (first team in schedule)
- `team_2` = `row['team2']` (second team in schedule)

**Example for Mercyhurst vs West Virginia:**
- Schedule shows: "Mercyhurst vs West Virginia"
- → team_1 = "Mercyhurst"
- → team_2 = "West Virginia"

## Step 2: Feature Building (build_flat_df.py)
All team stats are collected and labeled based on team_1/team_2:
- Mercyhurst's stats → all team_1_* columns
- West Virginia's stats → all team_2_* columns

## Step 3: Odds Matching (build_ou_features.py, lines 185-187)
The code checks which team is which in the DATABASE:
```python
home_is_team_1 = (home_team_mapped == team_1)
away_is_team_1 = (away_team_mapped == team_1)
```

**For Mercyhurst vs West Virginia:**
- Database: home_team="West Virginia Mountaineers", away_team="Mercyhurst Lakers"
- After mapping: home_team="West Virginia", away_team="Mercyhurst"
- team_1="Mercyhurst", team_2="West Virginia"
- Result: `away_is_team_1=TRUE` (Mercyhurst is away team and team_1)

## Step 4: Odds Assignment (lines 265-275)
```python
if away_is_team_1 and ml_away is not None:
    ml_for_team_1.append(ml_away)  # 2400 → team_1
if away_is_team_1 and ml_home is not None:
    ml_for_team_2.append(ml_home)  # -10000 → team_2
```

**Result:**
- team_1 (Mercyhurst): +2400 odds
- team_2 (West Virginia): -10000 odds

## THE PROBLEM

The odds assignment is logically correct, BUT the **model predicts Mercyhurst (team_1) at 64.1% with +2400 odds (4% implied)** - which is backwards!

## Possible Root Causes

1. **Schedule order is inconsistent**
   - Maybe the schedule sometimes lists away_team first, sometimes lists home_team first
   - This would cause team_1 to sometimes be home, sometimes be away

2. **The schedule order doesn't match home/away**
   - The schedule might list teams alphabetically or in some other order
   - But the database odds are always stored as home_team, away_team
   - If this doesn't match, teams get swapped

3. **The model was trained on data with different team_1/team_2 assignment logic**
   - The training data might have used home_team=team_1 instead of schedule order

## How to Verify

Check if the schedule consistently lists teams in the same order as the database:
- Is team_1 always the away team or always the home team?
- Or does it vary by game?
- Or is it always in a specific order (e.g., alphabetical)?

