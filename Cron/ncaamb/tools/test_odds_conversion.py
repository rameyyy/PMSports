#!/usr/bin/env python3
"""Test odds conversion logic"""

import sys
sys.path.insert(0, r'C:\Users\crame\Documents\Sport-Website\PMSports\Cron\ncaamb\models\overunder')

from ou_feature_build_utils import american_to_decimal, decimal_to_american, average_american_odds

# Test case from features2021.csv first game
# avg_ml_team_1 = 102.0
# avg_ml_team_2 = -119.99999999999999 (approx -120)

print("=" * 80)
print("TESTING ODDS CONVERSION LOGIC")
print("=" * 80)

# Example 1: Air Force vs Seattle (from features2021.csv line 2)
print("\n### EXAMPLE 1: Air Force vs Seattle ###")
print("Training data: avg_ml_team_1 = 102.0, avg_ml_team_2 = -120.0")

# Let's say the sportsbooks had:
example_odds_team_1 = [100, 105, 100, 105]  # Underdog
example_odds_team_2 = [-110, -120, -125, -120]  # Favorite

print(f"\nIf sportsbooks had (team_1): {example_odds_team_1}")
print(f"If sportsbooks had (team_2): {example_odds_team_2}")

avg_team_1 = average_american_odds(example_odds_team_1)
avg_team_2 = average_american_odds(example_odds_team_2)

print(f"Calculated avg_ml_team_1: {avg_team_1}")
print(f"Calculated avg_ml_team_2: {avg_team_2}")

print("\n### CONVERSION FUNCTION TESTS ###")

# Test american_to_decimal
test_cases = [100, -110, -120, 105, 102]
print("\nAmerican to Decimal:")
for american in test_cases:
    decimal = american_to_decimal(american)
    print(f"  {american:5} -> {decimal:.4f}")

# Test decimal_to_american
print("\nDecimal to American:")
decimal_cases = [2.0, 1.909, 1.833, 2.05, 1.02]
for decimal in decimal_cases:
    american = decimal_to_american(decimal)
    print(f"  {decimal:.4f} -> {american:.2f}")

# Test round-trip conversion
print("\n### ROUND-TRIP CONVERSION TEST ###")
original_odds = [100, -110, -120, 105, 102, 2400, -10000]
for original in original_odds:
    decimal = american_to_decimal(original)
    back_to_american = decimal_to_american(decimal)
    print(f"  {original:6} -> {decimal:.4f} -> {back_to_american:7.2f}")
