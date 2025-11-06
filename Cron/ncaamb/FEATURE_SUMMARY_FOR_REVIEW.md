# O/U Model Feature Summary - Ready for Review

## Overview
Total Features: **297 features** across **2,031 college basketball games**

## Feature Categories

### 1. Game Identifiers (5 features)
- game_id, date, team_1, team_2
- actual_total (target variable)

### 2. Odds Data (9 features)
- `avg_ou_line` - Average over/under line across sportsbooks
- `avg_spread` - Average point spread
- `avg_over_odds`, `avg_under_odds` - American odds
- `implied_fav_score`, `implied_dog_score` - Implied team scores from spread
- `ou_line_variance`, `spread_variance` - Market disagreement signals
- `spread_ou_agreement` - Correlation between spread and O/U
- `num_books_with_ou` - How many books have data

### 3. Team Scoring History (20 features)
Rolling window averages for last 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, and all-time games:
- `team_1_score_last{N}`, `team_2_score_last{N}`
- Also captures `team_X_score_alltime`

### 4. Team Scoring Variance (60 features)
Rolling window variance for:
- Score variance (shows scoring volatility)
- eFG% offensive variance (shooting consistency)
- Pace variance (tempo volatility)
All for windows: 1, 2, 3, 4, 5, 7, 9, 11, 13, 15

### 5. Team Scoring Trend (60 features)
Linear regression slope (improvement/decline) for:
- Score trends
- eFG% offensive trends
- Pace trends
All for windows: 1, 2, 3, 4, 5, 7, 9, 11, 13, 15

### 6. Team Efficiency Metrics (56 features)
- `team_X_efg_off_last{N}` - Effective FG% (all windows)
- `team_X_pace_last{N}` - Pace/possessions (all windows)
- `team_X_top5_efg_pct_last{N}` - Top 5 players shooting %
- `efg_off_differential`, `efg_def_differential` - Matchup efficiency differences

### 7. Rank-Based Historical Matchups (10 features)
For games vs opponents with similar rank:
- `team_X_score_closest3rank` - Avg score vs similar rank
- `team_X_score_allowed_closest3rank` - Avg points allowed
- `team_X_margin_closest3rank` - Point margin
- `team_X_total_closest3rank` - Average total points
- `combined_expected_total_closest_rank` - Combined prediction

### 8. Leaderboard Differentials (6 features)
- `rank_differential` - Rank difference
- `barthag_differential` - Power rating difference
- `adj_tempo_differential` - Adjusted tempo difference
- `ftr_differential` - Free throw rate difference
- `orb_rate_differential` - Offensive rebound rate difference
- `tor_rate_differential` - Turnover rate difference
- `3p_rate_differential` - 3-point rate difference
- `adj_oe_vs_opp_de` - Offensive efficiency vs opponent defense
- `adj_oe_opp_vs_de` - Opponent's offense vs our defense

### 9. Player Aggregation (60 features)
Top 5 players aggregated for windows: 1, 2, 3, 5, 7 games
- `team_X_top5_ppg_last{N}` - Points per game
- `team_X_top5_bpm_last{N}` - Box plus-minus
- `team_X_top5_usage_last{N}` - Usage rate
- `team_X_top5_efg_pct_last{N}` - Effective FG%
- `team_X_top5_ast_rate_last{N}` - Assist rate
- `team_X_top5_to_rate_last{N}` - Turnover rate
- `team_X_top5_min_per_last{N}` - Minutes per game

### 10. Data Quality (4 features)
- `team_X_games_available` - How many games in history
- `team_X_data_quality` - Confidence score (0-1) based on data completeness

### 11. Other (7 features)
- `team_X_score_alltime` - All-time average
- `num_books_with_ou` - Data availability indicator

---

## Feature Engineering Philosophy

### What Makes These Features?

1. **Rolling Windows**: 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, all-time games
   - Recent trends matter more than ancient history
   - Multiple windows capture different timeframes

2. **Three Metrics Per Stat**:
   - Average (central tendency)
   - Variance (volatility/consistency)
   - Trend (improving/declining)

3. **Matchup Specificity**:
   - Historical games vs similar ranked opponents
   - Four-factor differentials (efficiency metrics)

4. **Market Integration**:
   - Vegas consensus (avg_ou_line)
   - Sportsbook disagreement (variance/agreement signals)
   - Implied scores

5. **Player-Level Data**:
   - Top 5 players aggregated (high-variance contributors)
   - Per-game, BPM, efficiency metrics

---

## Questions to Ask About Missing Features

Based on common O/U models, here are features you might consider:

### 1. **Injury/Roster Changes**
- Key player absence flags
- Backups' average performance
- Currently: NO (would need real-time injury data)

### 2. **Home/Away Split**
- Home court advantage metrics
- Currently: NO (treating all games equally)

### 3. **Rest Days**
- Games on back-to-backs
- Days between games
- Currently: NO (would need game scheduling)

### 4. **Line Movement**
- How much Vegas line moved
- Direction of sharp action
- Currently: NO (using only current line)

### 5. **Conference-Level Features**
- Conference strength metrics
- Cross-conference tendencies
- Currently: NO (implicit in opponent rank)

### 6. **Advanced Possession Metrics**
- 4-factor differentials (we have some)
- Defensive rebounding %
- Free throw rate (we have this)
- Currently: PARTIAL (have FTR, ORB, TOR, 3P differentials)

### 7. **Score Range Buckets**
- High-scoring team tendencies (>80 ppg)
- Low-scoring team tendencies (<60 ppg)
- Currently: NO (using continuous scores)

### 8. **Opponent Strength Schedule**
- Average opponent quality faced
- SOS rating
- Currently: NO (implicit in leaderboard metrics)

### 9. **Under the Rim Features**
- Position-specific (guard vs center heavy)
- Pace-tempo relationship to scoring
- Currently: NO (only have pace)

### 10. **Model-Predicted Metrics**
- ML predictions of individual game outcomes
- Expected score from separate spread model
- Currently: NO (single model approach)

---

## Recommendations

### Keep (Core Features)
- All odds data (Vegas consensus is strong signal)
- Rolling window scores + variance + trend
- Four-factor differentials
- Player aggregation (top 5 captures star power)
- Rank-based matchups

### Add If Available
1. **Back-to-back indicator** (easy, just check dates)
2. **Key injury flags** (need real-time data integration)
3. **Line movement** (need historical Vegas data)
4. **Home/away splits** (easy, just track in data)

### Consider But Lower Priority
- Advanced position analysis (complex, diminishing returns)
- Score range buckets (trading complexity for accuracy)
- Opponent SOS (already captured by leaderboard metrics)

---

## Data Quality Notes

### Columns with String dtype (potential issue)
Many variance/trend/efficiency columns are stored as String instead of Float64.
This is likely from NaN handling and conversion. These should be numeric.

**FIX**: Convert String columns to Float64 before feeding to model:
```python
for col in features.columns:
    if features[col].dtype == String:
        features = features.with_columns(
            pl.col(col).cast(pl.Float64, strict=False)
        )
```

### Sparse Features
Some features may have many NaN values (teams with limited history).
Data quality metrics (team_X_data_quality) help weight these appropriately.

---

## Ready for Review

You can copy this summary and ask:
"Are these features good for O/U prediction? What am I missing that would improve accuracy?"

The model currently achieves:
- **Test MAE: 8.60 points** (very good)
- **68.5% within ±10 pts** (reliable for betting)

