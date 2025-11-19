
import polars as pl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.overunder.build_ou_features import build_ou_features

input_file = f"sample2022.parquet"
output_file = f"features2022.csv"

print(f"1. Loading {input_file}...")
flat_df = pl.read_parquet(input_file)
print(f"   Loaded {len(flat_df)} games")

print(f"\n2. Building O/U features with correct odds mapping...")
try:
    features_df = build_ou_features(flat_df)
    print(f"   Built features: {len(features_df)} rows, {len(features_df.columns)} columns")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n3. Saving to {output_file}...")
features_df.write_csv(output_file)
print(f"   [+] Saved to {output_file}")

print(f"\n4. Sample row:")
print(f"   game_id: {features_df['game_id'][0]}")
print(f"   date: {features_df['date'][0]}")
print(f"   team_1: {features_df['team_1'][0]}")
print(f"   team_2: {features_df['team_2'][0]}")
print(f"   actual_total: {features_df['actual_total'][0]}")
