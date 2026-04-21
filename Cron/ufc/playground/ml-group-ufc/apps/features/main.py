from pathlib import Path
from extract import get_fightsnapshots_df, unnest_raw_df
from build_features import FightFeatures

OUT = Path(__file__).parent.parent.parent / "data" / "features.csv"

def main():
    # 1. Extract and unnest raw data into 3 dfs that can be grouped by root_fight_id
    raw_df = get_fightsnapshots_df()
    fights_df, prior_fights_df, prior_rounds_df = unnest_raw_df(raw_df)

    # 2. Generate features
    features = FightFeatures(fights_df, prior_fights_df, prior_rounds_df)
    features.extract_fights_features()
    features.extract_prior_fights_features()
    features.final_df.write_csv(OUT)
    print(f"Wrote {features.final_df.shape[0]:,} rows x {features.final_df.shape[1]} cols to {OUT}")

if __name__ == "__main__":
    main()
