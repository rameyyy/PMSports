"""
Shared data loading and prep for March Madness bracket models.
Imported by each individual model training script.
"""

from pathlib import Path
import numpy as np
import polars as pl

ncaamb_dir = Path(__file__).parent.parent  # ncaamb/

# ── Odds columns to drop (not available at bracket time) ──────────────────────
ODDS_COLS = [
    # Per-book: BetMGM
    'betmgm_ou_line', 'betmgm_over_odds', 'betmgm_under_odds',
    'betmgm_ml_team_1', 'betmgm_ml_team_2',
    'betmgm_spread_pts_team_1', 'betmgm_spread_odds_team_1',
    'betmgm_spread_pts_team_2', 'betmgm_spread_odds_team_2',
    # Per-book: BetOnline
    'betonline_ou_line', 'betonline_over_odds', 'betonline_under_odds',
    'betonline_ml_team_1', 'betonline_ml_team_2',
    'betonline_spread_pts_team_1', 'betonline_spread_odds_team_1',
    'betonline_spread_pts_team_2', 'betonline_spread_odds_team_2',
    # Per-book: Bovada
    'bovada_ou_line', 'bovada_over_odds', 'bovada_under_odds',
    'bovada_ml_team_1', 'bovada_ml_team_2',
    'bovada_spread_pts_team_1', 'bovada_spread_odds_team_1',
    'bovada_spread_pts_team_2', 'bovada_spread_odds_team_2',
    # Per-book: DraftKings
    'draftkings_ou_line', 'draftkings_over_odds', 'draftkings_under_odds',
    'draftkings_ml_team_1', 'draftkings_ml_team_2',
    'draftkings_spread_pts_team_1', 'draftkings_spread_odds_team_1',
    'draftkings_spread_pts_team_2', 'draftkings_spread_odds_team_2',
    # Per-book: FanDuel
    'fanduel_ou_line', 'fanduel_over_odds', 'fanduel_under_odds',
    'fanduel_ml_team_1', 'fanduel_ml_team_2',
    'fanduel_spread_pts_team_1', 'fanduel_spread_odds_team_1',
    'fanduel_spread_pts_team_2', 'fanduel_spread_odds_team_2',
    # Per-book: LowVig
    'lowvig_ou_line', 'lowvig_over_odds', 'lowvig_under_odds',
    'lowvig_ml_team_1', 'lowvig_ml_team_2',
    'lowvig_spread_pts_team_1', 'lowvig_spread_odds_team_1',
    'lowvig_spread_pts_team_2', 'lowvig_spread_odds_team_2',
    # Per-book: MyBookie
    'mybookie_ou_line', 'mybookie_over_odds', 'mybookie_under_odds',
    'mybookie_ml_team_1', 'mybookie_ml_team_2',
    'mybookie_spread_pts_team_1', 'mybookie_spread_odds_team_1',
    'mybookie_spread_pts_team_2', 'mybookie_spread_odds_team_2',
    # Aggregate odds
    'avg_ou_line', 'ou_line_variance', 'avg_over_odds', 'avg_under_odds',
    'num_books_with_ou',
    'avg_spread_pts_team_1', 'avg_spread_pts_team_2', 'spread_variance',
    'avg_spread_odds_team_1', 'avg_spread_odds_team_2',
    'avg_ml_team_1', 'avg_ml_team_2',
    'num_books_with_spread', 'num_books_with_ml',
    # Derived odds features
    'implied_team_1_score', 'implied_team_2_score', 'spread_ou_agreement',
    'hours_until_game_from_odds',
    'combined_expected_total_closest3rank',
    'combined_expected_total_closest5rank',
    'combined_expected_total_closest7rank',
    # Game-time features (not available at bracket submission time)
    'is_early_game', 'is_afternoon_game', 'is_evening_game', 'is_late_game', 'hour_of_day',
]

METADATA_COLS = {
    'game_id', 'date', 'season', 'team_1', 'team_2',
    'team_1_score', 'team_2_score', 'actual_total',
    'team_1_conference', 'team_2_conference',
    'team_1_is_home', 'team_2_is_home', 'location',
    'start_time', 'game_odds', 'ou_target',
}


_INT_TYPES   = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_FLOAT_TYPES = (pl.Float32, pl.Float64)
_NUMERIC     = _INT_TYPES + _FLOAT_TYPES


def _normalize_schema(dfs: list) -> list:
    """
    Cast every non-metadata numeric column to Float64 across all dataframes
    so that diagonal concat never hits an Int/Float type mismatch.
    Columns that are String in one df but numeric elsewhere get cast to Float64
    (non-parseable values become null).
    """
    # Which columns are numeric in at least one df?
    numeric_cols = set()
    for df in dfs:
        for col in df.columns:
            if col not in METADATA_COLS and df[col].dtype in _NUMERIC:
                numeric_cols.add(col)

    normalized = []
    for df in dfs:
        casts = []
        for col in df.columns:
            if col in METADATA_COLS:
                continue
            if col in numeric_cols:
                if df[col].dtype not in _FLOAT_TYPES:
                    # Covers Int* of any width, and String-that-should-be-float
                    casts.append(pl.col(col).cast(pl.Float64, strict=False))
        if casts:
            df = df.with_columns(casts)
        normalized.append(df)
    return normalized


def load_all_features() -> pl.DataFrame:
    """Load features2021-2026.csv, drop odds cols, keep completed games only."""
    print("Loading features files...")
    dfs = []

    for year in range(2021, 2027):
        path = ncaamb_dir / f"features{year}.csv"
        if not path.exists():
            print(f"  ⚠  {path.name} not found — skipping")
            continue
        df = pl.read_csv(str(path), try_parse_dates=False, infer_schema_length=None)
        # Tag with source season so weighting uses season, not calendar year
        df = df.with_columns(pl.lit(str(year)).alias('_season'))
        dfs.append(df)
        print(f"  Loaded {path.name}: {len(df)} games")

    if not dfs:
        raise FileNotFoundError("No features*.csv files found in " + str(ncaamb_dir))

    dfs = _normalize_schema(dfs)

    # Build a unified target schema: Float64 for any column that is numeric
    # in at least one df; String otherwise. Then manually align all dfs.
    all_col_names = []
    seen = set()
    for df in dfs:
        for c in df.columns:
            if c not in seen:
                all_col_names.append(c)
                seen.add(c)

    # Determine target dtype for each column
    target_dtype: dict = {}
    for col in all_col_names:
        for df in dfs:
            if col in df.columns:
                if df[col].dtype in _NUMERIC:
                    target_dtype[col] = pl.Float64
                    break
        if col not in target_dtype:
            target_dtype[col] = pl.Utf8  # String fallback

    # Align every df: cast + fill missing columns
    aligned = []
    for df in dfs:
        casts = []
        for col in df.columns:
            tdtype = target_dtype.get(col)
            if tdtype and df[col].dtype != tdtype:
                casts.append(pl.col(col).cast(tdtype, strict=False))
        if casts:
            df = df.with_columns(casts)
        # Add any columns this df is missing
        for col in all_col_names:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(target_dtype[col]).alias(col))
        aligned.append(df.select(all_col_names))

    combined = pl.concat(aligned, how="vertical")
    print(f"  Combined: {len(combined)} total games")

    cols_to_drop = [c for c in ODDS_COLS if c in combined.columns]
    combined = combined.drop(cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} odds columns")

    # Keep completed games only (both scores must exist)
    combined = combined.filter(
        pl.col('team_1_score').is_not_null() &
        pl.col('team_2_score').is_not_null()
    )
    # Compute win target: 1 if team_1 wins, 0 if team_2 wins
    combined = combined.with_columns(
        (pl.col('team_1_score') > pl.col('team_2_score')).cast(pl.Int8).alias('team_1_win')
    )
    print(f"  Completed games: {len(combined)}\n")
    return combined


def get_feature_columns(df: pl.DataFrame) -> list:
    """Return numeric non-metadata, non-odds columns."""
    exclude = METADATA_COLS | set(ODDS_COLS) | {'team_1_win', 'actual_total', '_season'}
    return [
        col for col in df.columns
        if col not in exclude
        and df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8)
    ]


def _year_weights(season_array: np.ndarray, all_seasons: list) -> np.ndarray:
    """
    Linear recency weighting. all_seasons must be sorted ascending.
    Example for [2021,2022,2023,2024,2025,2026]:
      2021=1x, 2022=2x, 2023=3x, 2024=4x, 2025=5x, 2026=6x
    """
    season_to_weight = {s: float(i + 1) for i, s in enumerate(all_seasons)}
    return np.array([season_to_weight[s] for s in season_array])


def build_splits(df: pl.DataFrame, feature_cols: list):
    """
    Test = most recent season, train = all others.
    Linear sample weights: 2021=1x, 2022=2x, ..., 2026=6x.
    Returns X_train, y_train, X_test, y_test, sample_weights, seasons.
    """
    seasons = sorted(df['_season'].unique().to_list())
    most_recent = seasons[-1]

    train_df = df.filter(pl.col('_season') != most_recent)
    test_df  = df.filter(pl.col('_season') == most_recent)

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df['team_1_win'].to_numpy().astype(int)
    X_test  = test_df.select(feature_cols).fill_null(0).to_numpy()
    y_test  = test_df['team_1_win'].to_numpy().astype(int)

    train_seasons = [s for s in seasons if s != most_recent]
    weights = _year_weights(train_df['_season'].to_numpy(), train_seasons)

    print(f"  Seasons: {seasons[0]}–{seasons[-1]}")
    print(f"  Test  ({most_recent}): {len(X_test)} games")
    print(f"  Train: {len(X_train)} games")
    weight_str = "  Weights: " + ", ".join(
        f"{s}={i+1}x" for i, s in enumerate(train_seasons)
    )
    print(weight_str)
    return X_train, y_train, X_test, y_test, weights, seasons


def build_full_dataset(df: pl.DataFrame, feature_cols: list, seasons: list):
    """All data with linear season weights (2021=1x, 2022=2x, ..., 2026=6x)."""
    X_all = df.select(feature_cols).fill_null(0).to_numpy()
    y_all = df['team_1_win'].to_numpy().astype(int)

    weights_all = _year_weights(df['_season'].to_numpy(), seasons)
    weight_str = "  Full weights: " + ", ".join(
        f"{s}={i+1}x" for i, s in enumerate(seasons)
    )
    print(weight_str)
    return X_all, y_all, weights_all
