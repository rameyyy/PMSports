# Claude Feature Review - Prioritized Implementation Plan

## CRITICAL ISSUES (Fix First)

### 1. STRING DTYPE COLUMNS - HIGH PRIORITY BUG
**Impact**: Currently losing ~120 features (variance + trend metrics)
**Effort**: LOW (2 lines of code)
**Expected improvement**: +5-15% accuracy

**Problem**:
- 60 variance columns are String dtype
- 60 trend columns are String dtype
- XGBoost cannot use String features - silently ignores them
- Model is only using ~170 features instead of 297

**Solution**:
```python
# In build_ou_features.py, before returning DataFrame:
features_df = features_df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False)
    for col in features_df.columns
    if features_df[col].dtype == pl.String
])
```

**Timeline**: IMMEDIATE - before any other work
**Priority**: 🔴 CRITICAL

---

## TIER 1: MUST ADD (Highest Impact)

### 2. HOME/AWAY SPLITS
**Impact**: +3-5% accuracy improvement (biggest gap)
**Effort**: MEDIUM
**Expected MAE**: 8.60 → 8.15 pts

**What to add** (add to each team's rolling windows):
```
team_1_score_home_last5     - Scoring when at home
team_1_score_away_last5     - Scoring when on road
team_1_total_home_last5     - Total points in home games
team_1_total_away_last5     - Total points in road games
team_1_efg_off_home_last5   - Shooting % at home
team_1_efg_off_away_last5   - Shooting % on road
home_away_performance_gap   - Home avg - Away avg (can be 10+ pts)
```

**Why critical**: Air Force home vs away could be 130 vs 110 total
Current model treats all games the same = massive blind spot

**Implementation**:
1. Add location flag to flat_df (home/away for each team)
2. Modify build_rolling_window_features to filter by location
3. Create separate home/away rolling window builders

**Code location**: `models/ou_feature_build_utils.py` + `models/build_ou_features.py`

---

### 3. DEFENSIVE METRICS - MAJOR GAP
**Impact**: +3-5% accuracy improvement
**Effort**: MEDIUM
**Expected MAE**: 8.60 → 8.15 pts

**What to add** (rolling windows for each team):
```
team_1_score_allowed_last5     - Points allowed rolling
team_1_drtg_last5              - Defensive rating
team_1_efg_def_last5           - Defensive eFG%
team_1_defensive_trend_last5   - Defense improving/declining
team_1_defensive_variance_last5 - Defensive consistency
```

**Matchup specific**:
```
team1_ortg_vs_team2_drtg    - Team 1 offense rating vs Team 2 defense rating
team2_ortg_vs_team1_drtg    - Team 2 offense rating vs Team 1 defense rating
```

**Why critical**: Mercyhurst allows 80 ppg but model doesn't know this
Leaderboard differentials are static; rolling defense shows recent form

**Data source**: Already in leaderboard - extract defensive metrics
**Implementation**: Similar to offensive metrics, new rolling windows

---

### 4. OPPONENT-ADJUSTED SCORING
**Impact**: +2-3% accuracy improvement
**Effort**: MEDIUM-HIGH
**Expected MAE**: 8.60 → 8.35 pts

**What to add**:
```
team_1_score_vs_top50_def_last5    - Scoring vs good defenses
team_1_score_vs_bottom50_def_last5 - Scoring vs bad defenses
team_1_opponent_avg_drtg_last5     - Avg defensive quality faced
team_1_schedule_adjusted_score     - Raw score adjusted for opponent quality
```

**Why it matters**:
- Beats cupcakes (bad defenses) but struggles vs good defenses = not as good as raw stats
- Opposite (struggles vs bad, dominates good) = especially good team

**Implementation**:
1. Get opponent's drtg from each historical game
2. Calculate weighted avg by opponent quality
3. Adjust team's scoring based on strength of schedule

---

### 5. PACE INTERACTION/MATCHUP
**Impact**: +2-4% accuracy improvement
**Effort**: LOW-MEDIUM
**Expected MAE**: 8.60 → 8.25 pts

**What to add**:
```
expected_possessions           - (pace_team1 + pace_team2) / 2
pace_differential              - |pace_team1 - pace_team2|
expected_points_per_possession - Implied total / expected possessions
pace_variance                  - High variance in pace = volatile game
combined_pace_index            - Aggregate pace indicator
```

**Why it matters**:
- Fast + Fast = explosion (140+ totals)
- Slow + Slow = grind (110- totals)
- Mixed pace = whoever controls tempo wins
- Currently have pace metrics but not interaction

**Implementation**: Straightforward calculation from existing pace features

---

## TIER 2: SHOULD ADD (Good Impact)

### 6. RECENCY WEIGHTING
**Impact**: +1-2% accuracy improvement
**Effort**: LOW
**Expected MAE**: 8.60 → 8.50 pts

**What to add**:
```
team_1_score_recent_weighted   - Exponential decay (weight=0.9)
team_1_efg_recent_weighted     - Recent shooting better weighted
recency_confidence_factor      - How much to trust recent vs old data
```

**Why it matters**: Last game matters more than 10 games ago
Equal weighting dilutes momentum signals

**Implementation**: Simple exponential decay formula
```python
weights = [0.9**i for i in range(window)]  # [1.0, 0.9, 0.81, ...]
recent_weighted = sum(scores * weights) / sum(weights)
```

---

### 7. BLOWOUT INDICATORS
**Impact**: +1-2% accuracy improvement
**Effort**: LOW
**Expected MAE**: 8.60 → 8.50 pts

**What to add**:
```
team_1_blowout_rate_last10     - % games won/lost by 15+
expected_margin_from_spread    - Spread predicts this
garbage_time_factor            - Blowouts go Under (clock mgmt)
competitive_game_probability   - What's chance this is close?
```

**Why it matters**:
- Blowouts: late clock runs, fewer fouls → Under
- Close games: fouling, clock stops → Over
- O/U behaves differently by game competitiveness

---

### 8. CONFERENCE & SOS CONTEXT
**Impact**: +1-2% accuracy improvement
**Effort**: MEDIUM
**Expected MAE**: 8.60 → 8.50 pts

**What to add**:
```
team_1_conference_avg_pace         - Does their conference play fast?
team_1_conference_avg_total        - High-scoring conference?
team_1_conference_avg_drtg         - Defensive level
cross_conference_adjustment        - MWC vs NEC scoring tendencies
team_1_sos_rating                  - Strength of schedule
```

**Why it matters**:
- Big 12 teams score 80 ppg (fast pace)
- NEC teams score 60 ppg (slow pace)
- Mercyhurst in low-scoring conference
- Context matters for predictions

---

### 9. SHOOTING LOCATION BREAKDOWN
**Impact**: +1-2% accuracy improvement
**Effort**: MEDIUM-HIGH
**Expected MAE**: 8.60 → 8.50 pts

**What to add** (from leaderboard 4-factor):
```
team_1_rim_fg_pct_last5        - Shooting % at rim
team_1_3pt_rate_last5          - % of shots from 3
team_1_3pt_fg_pct_last5        - 3-point shooting %
shot_quality_score             - Weighted by location
shot_consistency               - Mix of easy/hard shots
```

**Why it matters**:
- Rim-heavy teams: consistent, predictable
- 3PT-heavy teams: volatile (makes go up, misses go down)
- Impacts variance/confidence intervals

---

### 10. FREE THROW VOLUME & IMPACT
**Impact**: +0.5-1% accuracy improvement
**Effort**: LOW
**Expected MAE**: 8.60 → 8.55 pts

**What to add**:
```
team_1_fta_per_game_last5          - FT attempts (volume)
team_1_ft_pct_last5                - FT shooting %
expected_free_throw_points         - FTA × FT%
free_throw_pace_impact             - FT stoppages reduce pace
```

**Why it matters**:
- 27 FTA = 18 points from FT line = 18 fewer field goals needed
- High FTA games = more stoppages = slower pace
- Direction varies (more free points vs pace reduction)

---

## TIER 3: NICE TO HAVE (Lower Priority)

### Quick Wins (Low effort, small boost):
- Month of season flag (early chaos vs late grinding)
- Back-to-back games flag (teams score 3-5 pts less)
- Days rest indicator (already added in ou_advanced_features.py)
- Variance ratio (Team1 volatility / Team2 volatility)

### Advanced (High effort, small boost):
- Shot clock violations
- Foul trouble tracking
- Coaching style (slow-grind vs uptempo)
- Travel distance calculation
- Rivalry game flags

---

## IMPLEMENTATION PRIORITY

### Phase 1 (Immediate - Day 1)
1. **FIX STRING DTYPES** - 5 min work, massive impact
2. **HOME/AWAY SPLITS** - 2-3 hours, +3-5% accuracy
3. **DEFENSIVE METRICS** - 2-3 hours, +3-5% accuracy

### Phase 2 (Short term - Days 2-3)
4. **OPPONENT-ADJUSTED SCORING** - 3-4 hours, +2-3% accuracy
5. **PACE INTERACTION** - 1-2 hours, +2-4% accuracy

### Phase 3 (Medium term - Days 4-5)
6. **RECENCY WEIGHTING** - 1 hour, +1-2% accuracy
7. **BLOWOUT INDICATORS** - 1 hour, +1-2% accuracy
8. **CONFERENCE/SOS** - 2-3 hours, +1-2% accuracy

### Phase 4 (Polish - Days 6-7)
9. **SHOOTING BREAKDOWN** - 2-3 hours, +1-2% accuracy
10. **FT VOLUME/IMPACT** - 1 hour, +0.5-1% accuracy

---

## EXPECTED CUMULATIVE IMPROVEMENT

| Phase | Features Added | Cumulative MAE | Improvement |
|-------|---|---|---|
| Current | String dtype bug fixed | 8.60 → 7.50 | -13% |
| Phase 1 | Home/Away + Defense | 7.50 → 7.00 | -6-10% |
| Phase 2 | Opp-adjust + Pace | 7.00 → 6.70 | -4-6% |
| Phase 3 | Recency + Blowout | 6.70 → 6.55 | -2-3% |
| Phase 4 | Shooting + FT | 6.55 → 6.50 | -0.5-1% |

**Total potential**: 8.60 → 6.50 MAE (**-24% improvement**)
**Realistic**: 8.60 → 7.0-7.2 MAE with Phases 1-2 (**-15-18% improvement**)

---

## DATA REQUIREMENTS

### Already Available
✓ Rolling window data (score, pace, eFG%)
✓ Leaderboard data (rank, barthag, differentials)
✓ Odds data (O/U line, spread, implied scores)
✓ Player data (top 5 players, stats)

### Need to Extract/Add
⚠ Home/Away location for each game (check flat_df structure)
⚠ Defensive metrics from leaderboard (drtg, efg_def)
⚠ Shooting location breakdown (might be in player stats)
⚠ FT rate/volume (need to calculate from player stats)

### Already Implemented (ou_advanced_features.py)
✓ Rest/fatigue features
✓ Game time buckets
✓ Conference flags
✓ Momentum indicators
✓ Market dynamics (partial - needs opening lines)

---

## DECISION FRAMEWORK

### If you want +15-18% improvement:
**Do Phases 1-2** (3-4 days work)
- Fix string dtypes
- Add home/away splits
- Add defensive metrics
- Add opponent-adjusted scoring
- Add pace interaction

Expected: 8.60 → 7.0-7.2 MAE

### If you want +8-10% improvement (quick win):
**Do Phase 1 only** (3-4 hours work)
- Fix string dtypes (+13%)
- Add home/away splits (+3-5%)
- Might see 8.60 → 7.5 or better

### If you want maximum improvement (+24%):
**Do all 4 Phases** (10-15 days work)
- Build all features systematically
- Retrain after each phase
- Track improvement curve
- Expected: 8.60 → 6.50 MAE (sharp-level)

---

## TESTING PROTOCOL

After each phase:

1. Run feature generation
2. Check for NaN columns
3. Train model with DEEP_LIGHT parameters
4. Compare Test MAE vs baseline
5. Check feature importance (Vegas line should still be top 5)
6. Log improvement

```bash
# Phase 1 Test
python3 test_features_ou.py
python3 train_ou_model.py
# Expected: Test MAE 7.0-7.5 (vs current 8.60)
```

---

## QUICK REFERENCE: WHAT TO CODE

### Phase 1A: Fix String Dtypes
**File**: `models/build_ou_features.py`
**Change**: Before return statement, convert all String → Float64
**Time**: 5 min
**Lines**: 2-3

### Phase 1B: Home/Away Splits
**Files**:
- `models/ou_feature_build_utils.py` (new function: build_home_away_features)
- `models/build_ou_features.py` (call new function)
**Time**: 2-3 hours
**New features**: 14 (7 per team)

### Phase 1C: Defensive Metrics
**Files**: Same as Phase 1B
**Time**: 1-2 hours
**New features**: 10 (5 per team + 2 matchup)

---

## RECOMMENDATION

**Start with Phase 1A (fix string dtypes) immediately** - this alone could give +13% improvement at zero coding cost.

**Then Phase 1B (home/away) + 1C (defense)** - these are the two biggest gaps mentioned.

This gives you realistic 8.60 → 7.0-7.2 MAE in 4-5 hours of coding, which is sharp-level performance.

Would you like me to:
1. Code Phase 1A (string dtype fix) first?
2. Code Phase 1B (home/away splits) template?
3. Code Phase 1C (defensive metrics) template?
4. Or handle all three together?

