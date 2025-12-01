
#!/usr/bin/env python3
"""
Regenerate all features files for years 2021-2025 with new odds mapping
"""
import subprocess
import sys
import os

def regenerate_features(year):
    """Regenerate features for a given year"""
    print(f"\n{'='*80}")
    print(f"REGENERATING FEATURES FOR {year}")
    print(f"{'='*80}\n")

    script = f"""
import polars as pl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.overunder.build_ou_features import build_ou_features

input_file = f"sample{year}.parquet"
output_file = f"features{year}.csv"

print(f"1. Loading {{input_file}}...")
flat_df = pl.read_parquet(input_file)
print(f"   Loaded {{len(flat_df)}} games")

print(f"\\n2. Building O/U features with correct odds mapping...")
try:
    features_df = build_ou_features(flat_df)
    print(f"   Built features: {{len(features_df)}} rows, {{len(features_df.columns)}} columns")
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\\n3. Saving to {{output_file}}...")
features_df.write_csv(output_file)
print(f"   [+] Saved to {{output_file}}")

print(f"\\n4. Sample row:")
print(f"   game_id: {{features_df['game_id'][0]}}")
print(f"   date: {{features_df['date'][0]}}")
print(f"   team_1: {{features_df['team_1'][0]}}")
print(f"   team_2: {{features_df['team_2'][0]}}")
print(f"   actual_total: {{features_df['actual_total'][0]}}")
"""

    # Write temp script
    temp_file = f'_regen_{year}.py'
    with open(temp_file, 'w') as f:
        f.write(script)

    try:
        result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"ERROR: Features generation failed for {year}")
            return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return True


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    years = ['2021', '2022', '2023', '2024', '2025']
    failed_years = []

    for year in years:
        if not regenerate_features(year):
            failed_years.append(year)

    print(f"\n{'='*80}")
    if failed_years:
        print(f"FAILED YEARS: {', '.join(failed_years)}")
        return 1
    else:
        print("ALL FEATURES REGENERATED SUCCESSFULLY!")
        print(f"{'='*80}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
