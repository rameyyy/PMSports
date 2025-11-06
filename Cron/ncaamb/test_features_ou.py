#!/usr/bin/env python3
"""
Test Over/Under feature engineering
"""
import polars as pl
from models.build_ou_features import build_ou_features

if __name__ == "__main__":
    print("Loading sample data...")
    df = pl.read_parquet("sample2.parquet")
    print(f"Loaded {len(df)} games\n")

    print("Building O/U features...")
    ou_features = build_ou_features(df)

    print(f"\nO/U Features shape: {ou_features.shape}")
    print(f"Columns: {ou_features.columns}\n")

    # Show feature count
    print(f"Total features generated: {len(ou_features.columns)}")
    print(f"Features per game: {len(ou_features.columns) - 4}")  # Excluding game_id, date, team_1, team_2

    # Don't print full dataframe due to unicode issues, just confirm data exists
    print("Sample game data loaded - checking for missing values...")
    numeric_cols = [col for col in ou_features.columns if col not in ['game_id', 'date', 'team_1', 'team_2']]
    null_counts = ou_features.select(numeric_cols).null_count()
    print(f"Null values in feature set - checking complete...")

    # FIX: Convert String dtypes to Float64, but preserve identifier columns as Utf8
    print("\nConverting String dtypes to Float64...")
    identifier_cols = {'game_id', 'date', 'team_1', 'team_2'}
    string_cols = [col for col in ou_features.columns if ou_features[col].dtype == pl.String and col not in identifier_cols]
    print(f"  Found {len(string_cols)} String columns to convert (excluding identifiers)")

    if string_cols:
        ou_features = ou_features.with_columns([
            pl.col(col).cast(pl.Float64, strict=False).alias(col)
            for col in string_cols
        ])
        print(f"  Converted {len(string_cols)} columns to Float64")

    # Reorder columns: game_id first, then all features
    cols = ou_features.columns
    reordered = ['game_id'] + [c for c in cols if c != 'game_id']
    ou_features = ou_features.select(reordered)

    # Save to CSV
    print("\n" + "="*80)
    print("Saving to CSV")
    print("="*80)
    import os
    import shutil

    csv_path = "ou_features.csv"
    temp_path = "ou_features_temp.csv"

    # Write to temp file first
    ou_features.write_csv(temp_path)

    # Replace original file
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except:
            pass

    shutil.move(temp_path, csv_path)
    print(f"Saved to {csv_path}")

    print("\nComplete!")
