#!/usr/bin/env python3
"""
Analyze features built from sample.parquet - check for nulls and data quality
"""
import polars as pl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.overunder.build_ou_features import build_ou_features

print("="*90)
print("BUILDING AND ANALYZING FEATURES FROM sample.parquet")
print("="*90)

# Load sample.parquet
print("\n1. Loading sample.parquet...")
flat_df = pl.read_parquet("sample.parquet")
print(f"   Loaded {len(flat_df)} games")

# Build features
print("\n2. Building O/U features...")
features_df = build_ou_features(flat_df)
print(f"   ✅ Built features: {len(features_df)} rows, {len(features_df.columns)} columns")

# Save features
features_df.write_csv("ou_features.csv")
print("   ✅ Saved to ou_features.csv")

# Group features by category
print("\n3. Categorizing features...")
feature_categories = {
    'adjoe': [],
    'adjde': [],
    'odds': [],
    'rolling_windows': [],
    'closest_rank': [],
    'leaderboard_diff': [],
    'other': []
}

for col in features_df.columns:
    col_lower = col.lower()
    if 'adjoe' in col_lower:
        feature_categories['adjoe'].append(col)
    elif 'adjde' in col_lower:
        feature_categories['adjde'].append(col)
    elif any(x in col_lower for x in ['ou_line', 'over_odds', 'under_odds', 'ml_', 'spread', 'betmgm', 'fanduel', 'draftkings']):
        feature_categories['odds'].append(col)
    elif 'closest' in col_lower and 'rank' in col_lower:
        feature_categories['closest_rank'].append(col)
    elif 'last' in col_lower and any(x in col_lower for x in ['2', '3', '4', '5', '9', '11', '13', '15']):
        feature_categories['rolling_windows'].append(col)
    elif 'differential' in col_lower or 'combined' in col_lower:
        feature_categories['leaderboard_diff'].append(col)
    else:
        feature_categories['other'].append(col)

# Calculate stats for each category
print("\n📊 NULL STATISTICS BY CATEGORY")
print("="*90)

category_stats = []
for category, cols in feature_categories.items():
    if not cols:
        continue

    null_counts = [features_df[col].null_count() for col in cols]
    total = len(features_df)
    avg_null_pct = sum((n/total*100) for n in null_counts) / len(null_counts) if null_counts else 0
    max_null_pct = max((n/total*100) for n in null_counts) if null_counts else 0
    min_null_pct = min((n/total*100) for n in null_counts) if null_counts else 0

    category_stats.append({
        'category': category,
        'num_features': len(cols),
        'avg_null_pct': avg_null_pct,
        'min_null_pct': min_null_pct,
        'max_null_pct': max_null_pct
    })

# Sort by average null percentage
category_stats.sort(key=lambda x: x['avg_null_pct'], reverse=True)

print(f"\n{'Category':<20} {'# Features':<12} {'Avg Null %':<12} {'Min %':<10} {'Max %':<10}")
print("-"*70)
for stat in category_stats:
    print(f"{stat['category']:<20} {stat['num_features']:<12} {stat['avg_null_pct']:>10.2f}% {stat['min_null_pct']:>8.1f}% {stat['max_null_pct']:>8.1f}%")

# Overall statistics
print("\n" + "="*90)
print("OVERALL STATISTICS")
print("="*90)

total_rows = len(features_df)
total_cols = len(features_df.columns)
total_cells = total_rows * total_cols

# Calculate total nulls
total_nulls = 0
for col in features_df.columns:
    total_nulls += features_df[col].null_count()

overall_null_pct = (total_nulls / total_cells * 100) if total_cells > 0 else 0

print(f"\nDataset:")
print(f"  Rows (games): {total_rows:,}")
print(f"  Columns (features): {total_cols:,}")
print(f"  Total cells: {total_cells:,}")
print(f"  Total nulls: {total_nulls:,}")
print(f"  Overall null %: {overall_null_pct:.2f}%")

# Identify columns with highest nulls
null_data = []
for col in features_df.columns:
    null_count = features_df[col].null_count()
    null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
    null_data.append((col, null_count, null_pct))

null_data.sort(key=lambda x: x[2], reverse=True)

print("\n⚠️  TOP 15 COLUMNS WITH MOST NULLS:")
print("-"*90)
print(f"{'Column':<60} {'Null Count':<12} {'Null %':<10}")
print("-"*90)
for col, count, pct in null_data[:15]:
    print(f"{col:<60} {count:<12} {pct:>9.1f}%")

# Check if we have key required columns
print("\n✅ KEY FEATURE AVAILABILITY:")
print("-"*90)
key_features = [
    'actual_total', 'avg_ou_line',
    'team_1_adjoe', 'team_2_adjoe', 'team_1_adjde', 'team_2_adjde',
    'adjoe_differential', 'adjde_differential',
    'combined_adjoe', 'combined_adjde',
    'barthag_differential', 'adj_tempo_differential',
]

for feat in key_features:
    if feat in features_df.columns:
        null_count = features_df[feat].null_count()
        pct = (null_count / total_rows * 100)
        status = "✅" if pct < 5 else "⚠️" if pct < 20 else "❌"
        print(f"  {status} {feat:<35} {null_count:>5} nulls ({pct:>5.1f}%)")
    else:
        print(f"  ❌ {feat:<35} MISSING")

print("\n" + "="*90)
print("CONCLUSION")
print("="*90)

# Provide recommendations
high_null_features = [x for x in null_data if x[2] > 50]
medium_null_features = [x for x in null_data if 20 < x[2] <= 50]

if high_null_features:
    print(f"\n⚠️  {len(high_null_features)} features have >50% nulls - consider removing these")
elif medium_null_features:
    print(f"\n⚠️  {len(medium_null_features)} features have 20-50% nulls - might need investigation")
else:
    print("\n✅ All features have <20% nulls - data quality looks good!")

print(f"\n📊 adjoe features: {len(feature_categories['adjoe'])} columns, avg {sum((features_df[c].null_count()/total_rows*100) for c in feature_categories['adjoe'])/len(feature_categories['adjoe']):.1f}% nulls")
print(f"📊 adjde features: {len(feature_categories['adjde'])} columns, avg {sum((features_df[c].null_count()/total_rows*100) for c in feature_categories['adjde'])/len(feature_categories['adjde']):.1f}% nulls")

print("\n" + "="*90)
