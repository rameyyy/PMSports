#!/usr/bin/env python3
"""
Generate features and validate before training
"""
import polars as pl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.build_ou_features import build_ou_features

# ============================================================================
# CONFIGURATION - EDIT YEAR HERE
# ============================================================================
YEAR = "2025"  # Change this to desired year (e.g., "2021", "2022", "2023", or leave blank for "sample")
# =========================================oi===================================

print("="*80)
print("GENERATING FEATURES AND VALIDATING")
print("="*80)

# Determine input and output filenames based on year
if YEAR and YEAR.strip():
    input_file = f"sample{YEAR}.parquet"
    output_file = f"features{YEAR}.csv"
else:
    input_file = "sample.parquet"
    output_file = "features.csv"

# Load sample.parquet
print(f"\n1. Loading {input_file}...")
flat_df = pl.read_parquet(input_file)
print(f"   Loaded {len(flat_df)} games")

# Build features
print("\n2. Building O/U features...")
features_df = build_ou_features(flat_df)
print(f"   Built features: {len(features_df)} rows, {len(features_df.columns)} columns")

# Save features
print(f"\n3. Saving to {output_file}...")
features_df.write_csv(output_file)
print(f"   ✅ Saved to {output_file}")

# Show sample row with game_id and actual_total for validation
print("\n4. Sample row for validation:")
print("="*80)
sample = features_df.select(['game_id', 'date', 'team_1', 'team_2', 'actual_total']).head(1)
print(sample)

print("\n5. Checking for data leakage concerns...")
print("   Columns that will be EXCLUDED from features:")
exclude_cols = ['game_id', 'date', 'team_1', 'team_2', 'actual_total', 'team_1_score', 'team_2_score']
for col in exclude_cols:
    if col in features_df.columns:
        print(f"   ✅ {col} - will be excluded (not used as feature)")
    else:
        print(f"   ⚠️  {col} - not found in dataset")

print("\n6. Feature columns (first 20):")
numeric_cols = [c for c in features_df.columns if c not in exclude_cols]
for i, col in enumerate(numeric_cols[:20], 1):
    print(f"   {i}. {col}")
print(f"   ... and {len(numeric_cols) - 20} more features")

print("\n" + "="*80)
print("READY TO TRAIN MODEL")
print("="*80)
