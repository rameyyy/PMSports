#!/usr/bin/env python3
"""
Generate features2026.csv from sample2026.parquet.
Reads  ncaamb/sample2026.parquet
Writes ncaamb/features2026.csv

Must be run after build_sample2026.py.

Run from ncaamb/ directory:
    python marchmadness/generate_features2026.py
"""

import sys
from pathlib import Path
import polars as pl

# Add ncaamb/ to path so models/ package is importable
ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

from models.overunder.build_ou_features import build_ou_features

YEAR        = "2026"
INPUT_FILE  = ncaamb_dir / f"sample{YEAR}.parquet"
OUTPUT_FILE = ncaamb_dir / f"features{YEAR}.csv"

print("=" * 80)
print(f"GENERATING FEATURES{YEAR}.CSV")
print("=" * 80)

# ── 1. Load parquet ───────────────────────────────────────────────────────────
print(f"\n1. Loading {INPUT_FILE.name}...")
if not INPUT_FILE.exists():
    print(f"   ERROR: {INPUT_FILE} not found.")
    print("   Run build_sample2026.py first.")
    sys.exit(1)

flat_df = pl.read_parquet(str(INPUT_FILE))
print(f"   Loaded {len(flat_df)} games")

# ── 2. Build O/U features ─────────────────────────────────────────────────────
print("\n2. Building O/U features...")
features_df = build_ou_features(flat_df)
print(f"   Built features: {len(features_df)} rows, {len(features_df.columns)} columns")

# ── 3. Keep only completed games (actual_total not null) ─────────────────────
before = len(features_df)
features_df = features_df.filter(pl.col('actual_total').is_not_null())
print(f"   Completed games (actual_total not null): {len(features_df)} "
      f"(dropped {before - len(features_df)})")

# ── 4. Save ───────────────────────────────────────────────────────────────────
print(f"\n3. Saving to {OUTPUT_FILE.name}...")
features_df.write_csv(str(OUTPUT_FILE))
print(f"   ✅ Saved to {OUTPUT_FILE}")

# ── 5. Quick validation ───────────────────────────────────────────────────────
print("\n4. Sample rows:")
print(features_df.select(['game_id', 'date', 'team_1', 'team_2', 'actual_total']).head(3))

print(f"\n5. Column count: {len(features_df.columns)} total columns")
print("\nDone! Run train_bracket_model.py to train the March Madness model.")
