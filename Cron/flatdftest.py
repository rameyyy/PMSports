import polars as pl
from models.ufc_mma import flat_df_build, flat_df_to_model_ready
# flat_df_build.run()
# Load the data
df = pl.read_parquet('fight_snapshots.parquet')
trainingdf = flat_df_to_model_ready.create_differential_features(df)
trainingdf.write_csv('trainingset.csv')

# import pandas as pd

# # Load the data
# df = pd.read_parquet('fight_snapshots.parquet')

# # Calculate total fights before cleaning
# total_fights_original = len(df)

# # Count fighter1 wins
# f1_wins = (df['winner_id'] == df['fighter1_id']).sum()

# # Count fighter2 wins  
# f2_wins = (df['winner_id'] == df['fighter2_id']).sum()

# # Check for any fights where winner is neither f1 nor f2 (data issues)
# neither_wins = total_fights_original - f1_wins - f2_wins

# print(f"Original total fights: {total_fights_original:,}")
# print(f"Fighter1 wins: {f1_wins:,}")
# print(f"Fighter2 wins: {f2_wins:,}")
# print(f"Neither (data issues): {neither_wins:,}")

# if neither_wins > 0:
#     print(f"\n🔧 Cleaning data: removing {neither_wins:,} fights with data issues...")
    
#     # Create boolean mask for valid fights (winner is either f1 or f2)
#     valid_fights = (df['winner_id'] == df['fighter1_id']) | (df['winner_id'] == df['fighter2_id'])
    
#     # Filter to only valid fights
#     df_clean = df[valid_fights].copy()
    
#     # Save the cleaned dataset
#     df_clean.to_parquet('fight_snapshots.parquet')
    
#     print(f"✅ Saved cleaned dataset with {len(df_clean):,} fights")
    
#     # Recalculate stats with clean data
#     total_fights = len(df_clean)
#     f1_wins = (df_clean['winner_id'] == df_clean['fighter1_id']).sum()
#     f2_wins = (df_clean['winner_id'] == df_clean['fighter2_id']).sum()
    
# else:
#     print("\n✅ No data issues found - all fights have valid winners")
#     total_fights = total_fights_original

# # Calculate percentages with clean data
# f1_win_pct = (f1_wins / total_fights) * 100
# f2_win_pct = (f2_wins / total_fights) * 100

# print(f"\nFinal stats:")
# print(f"Total fights: {total_fights:,}")
# print(f"Fighter1 wins: {f1_wins:,} ({f1_win_pct:.1f}%)")
# print(f"Fighter2 wins: {f2_wins:,} ({f2_win_pct:.1f}%)")
# print()
# print(f"Bias check:")
# if abs(f1_win_pct - 50) > 5:  # More than 5% deviation from 50%
#     print(f"⚠️  BIAS DETECTED: Fighter1 wins {f1_win_pct:.1f}% of the time")
#     if f1_win_pct > 55:
#         print("   Fighter1 position seems to favor winners")
#     else:
#         print("   Fighter2 position seems to favor winners")
# else:
#     print(f"✅ No significant bias detected (Fighter1 wins {f1_win_pct:.1f}%)")