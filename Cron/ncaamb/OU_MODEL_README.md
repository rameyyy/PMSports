# Over/Under Prediction Model

Complete XGBoost-based pipeline for predicting basketball game totals (over/under lines).

## System Overview

```
test.py
  ↓ (builds flat dataset with match history, leaderboard, odds)
sample.parquet
  ↓
test_features_ou.py
  ↓ (computes 291 features from game data)
ou_features.csv
  ↓
train_ou_model.py
  ↓ (trains XGBoost model)
ou_model.pkl
  ↓
predict_ou.py / ou_pipeline.py
  ↓ (makes predictions on new games)
ou_predictions.csv
```

## Files

### Data Building
- **test.py** - Builds flat dataset with game data, match history, leaderboard stats, and odds
- **sample.parquet** - Cached flat dataset

### Feature Engineering
- **models/build_flat_df.py** - Builds flat dataframe with historical match data
- **models/build_ou_features.py** - Computes 291 predictive features
- **models/ou_feature_build_utils.py** - Helper functions for feature engineering
- **test_features_ou.py** - Test script to generate features and save to CSV
- **ou_features.csv** - Generated features (495 games × 291 features)

### Model Training & Prediction
- **models/ou_model.py** - XGBoost model class with train/predict/load/save methods
- **train_ou_model.py** - Training script that trains model and shows feature importance
- **predict_ou.py** - Prediction script that loads model and makes predictions
- **ou_pipeline.py** - End-to-end pipeline (build features + predict in one script)
- **ou_model.pkl** - Trained XGBoost model

### Outputs
- **ou_predictions.csv** - Model predictions on all games
- **ou_predictions_detailed.csv** - Predictions with additional metrics
- **ou_predictions_with_signals.csv** - Predictions with O/U signals and edges

## Quick Start

### 1. Build Dataset (if needed)
```bash
python3 test.py
# Generates: sample.parquet
```

### 2. Generate Features
```bash
python3 test_features_ou.py
# Generates: ou_features.csv (291 features × 495 games)
```

### 3. Train Model
```bash
python3 train_ou_model.py
# Generates: ou_model.pkl (trained model)
# Shows: Feature importance, training metrics
```

### 4. Make Predictions
```bash
# Option A: Use pre-computed features
python3 predict_ou.py

# Option B: End-to-end pipeline from DB
python3 ou_pipeline.py --rebuild --start 2025-02-01 --end 2025-02-15

# Option C: Use saved features with custom date range
python3 ou_pipeline.py --start 2025-02-01 --end 2025-02-15
```

## Model Performance

Training Results (from train_ou_model.py):
- **Training MAE**: 1.67 points (regularized, more honest)
- **Test MAE**: 12.25 points (20% holdout)
- **Training RMSE**: 2.59 points
- **Test RMSE**: 15.03 points
- **Train/Test Gap**: 10.58 points (reduced overfitting by 9%)

Prediction Results (all 495 games):
- **Mean Absolute Error**: 3.79 points
- **Over/Under Split**: 265 Over, 220 Under
- **Model Stability**: Improved generalization to unseen data

### Top 10 Most Important Features (After Regularization)
1. ftr_differential (115) - Free throw rate difference matters most
2. avg_ou_line (105) - Vegas consensus line is strong signal
3. team_1_score_trend_last7 (88) - Recent scoring trend (7 games)
4. team_2_score_trend_last4 (84) - Opponent's recent trend
5. team_1_score_trend_last4 (82) - Recent scoring trend (4 games)
6. implied_fav_score (76) - Spread + O/U combined signal
7. team_1_score_trend_last5 (74) - Recent scoring trend (5 games)
8. team_2_score_trend_last7 (73) - Opponent's 7-game trend
9. ou_line_variance (71) - Market disagreement on total
10. avg_spread (69) - Average point spread

## Feature Categories (291 Total)

### 1. Rolling Window Features (~180)
- Team scoring last 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, all-time games
- For each window: average, variance, and trend
- Stats: Points, eFG%, Pace
- Both teams

### 2. Rank-Based Historical Matchups (~16)
- Games from history with closest opponent rank
- Average scores, totals, and margins
- Combined expected total prediction

### 3. Leaderboard Differentials (~11)
- Rank differential
- Barthag power rating differential
- Adjusted tempo, offensive efficiency, defensive efficiency
- Four factors: eFG%, ORB%, Turnover rate, FTR differentials

### 4. Player Aggregation (~70)
- Top 5 players by minutes for windows: 1, 2, 3, 5, 7 games
- Per player: PPG, BPM, Usage, eFG%, Assist rate, Turnover rate, Minutes
- Both teams

### 5. Market Odds Data (~13)
- Vegas O/U line (average across 9 sportsbooks)
- Over/Under American odds
- Point spread
- Implied team scores from spread
- Market disagreement signals

### 6. Data Quality (~4)
- Games available in history
- Data quality confidence score (0-1)
- Allows model to weight unreliable games lower

## Model Details

### XGBoost Configuration (with Regularization)
```python
objective: 'reg:squarederror'  # Regression task
learning_rate: 0.05           # Slow learning to prevent overfitting
max_depth: 5                  # Shallower trees for better generalization
min_child_weight: 5           # Require more samples per leaf
subsample: 0.7                # Use 70% of data per tree (not 100%)
colsample_bytree: 0.7         # Use 70% of features per tree
reg_alpha: 1.0                # L1 regularization
reg_lambda: 1.0               # L2 regularization
n_estimators: 100
```

### Input
- 284 numeric features (after dropping identifiers)
- NaN values converted to 0
- No missing value imputation needed

### Output
- Single continuous value: predicted total points

### Predictions Include
- Game identifiers (game_id, date, team_1, team_2)
- Predicted total points
- Actual total points (if available)
- Prediction error
- O/U signal (Over/Under with edge)

## Usage Examples

### Load Trained Model and Predict
```python
from models.ou_model import OUModel
import polars as pl

# Load model
model = OUModel("ou_model.pkl")

# Load features
features_df = pl.read_csv("ou_features.csv")

# Make predictions
predictions = model.predict(features_df)

# Or predict single game
single_pred = model.predict_single(features_dict)
```

### Get Feature Importance
```python
importance = model.get_feature_importance(top_n=20)
for feat, score in importance.items():
    print(f"{feat}: {score}")
```

### Get O/U Signal
```python
predicted_total = 129.5
market_line = 128.0

ou_pred = model.get_ou_prediction(predicted_total, market_line)
print(ou_pred)
# {'predicted_total': 129.5, 'ou_signal': 'Over', 'market_line': 128.0, 'edge': 1.5}
```

## Data Flow

```
Raw Data (Database)
├─ Games (scores, stats, dates)
├─ Teams (leaderboard, ratings)
├─ Player Stats (per game)
└─ Odds (from sportsbooks)
       ↓
    test.py (build_flat_df.py)
       ↓
    sample.parquet (flat dataset with nested match history)
       ↓
    test_features_ou.py (build_ou_features.py)
       ↓
    ou_features.csv (291 numeric features)
       ↓
    train_ou_model.py
       ↓
    ou_model.pkl (trained XGBoost)
       ↓
    predict_ou.py / ou_pipeline.py
       ↓
    ou_predictions.csv (predictions + signals)
```

## Configuration

### Date Range
Modify in scripts:
```python
start_date = "2025-02-01"
end_date = "2025-02-15"
```

### Model Hyperparameters
In train_ou_model.py:
```python
metrics = model.train(
    features_df,
    test_size=0.2,
    learning_rate=0.05,      # Adjust learning rate
    max_depth=7,             # Tree depth
    n_estimators=150,        # Number of trees
)
```

### Sportsbooks
Currently averages odds from 9 sportsbooks (configurable in build_flat_df.py):
- avg_ou_line: average over/under line
- avg_over_odds / avg_under_odds: average american odds
- avg_spread: average point spread

## Troubleshooting

### "Model must be trained before making predictions"
- Train the model first: `python3 train_ou_model.py`

### Feature count mismatch
- Ensure ou_features.csv was generated with same build_ou_features.py
- Retrain model if features changed

### Low prediction accuracy
- Check data quality (are games recent and complete?)
- Validate that leaderboard data is available
- Retrain model with recent data

### Missing odds data
- Some games may not have odds from all sportsbooks
- Model handles missing values (converted to 0)
- Data quality metric reflects confidence

## Performance Notes

- **Training**: Fast (< 5 seconds on full dataset)
- **Prediction**: Very fast (< 1 second for 500 games)
- **Feature generation**: ~30 seconds for 500 games
- **Model file size**: ~570 KB

## Next Steps

1. **Backtesting**: Compare predictions against historical lines
2. **Odds Adjustment**: Weight by Vegas line movement
3. **Player Tracking**: Monitor key injuries and lineup changes
4. **Ensemble**: Combine with other models for confidence
5. **Betting Signals**: Add kelly criterion for bet sizing

## Dependencies

- polars
- xgboost
- numpy
- scikit-learn (optional, for advanced metrics)

Install with:
```bash
pip install polars xgboost numpy
```
