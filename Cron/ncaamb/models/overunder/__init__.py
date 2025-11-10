"""
Over/Under (OU) Modeling Module

Contains all logic for:
- Training and saving models (XGBoost, LightGBM, CatBoost)
- Loading pre-trained models
- Building and generating features for OU predictions
- Ensemble models combining multiple model predictions
"""

from .ou_model import OUModel
from .lgb_model import LGBModel
from .cat_model import CatModel
from .ensemble3_model import Ensemble3Model
from .build_ou_features import build_ou_features
from .ou_advanced_features import (
    calculate_rest_features,
    calculate_venue_features,
    calculate_conference_features,
    calculate_game_time_features,
)
from .ou_feature_build_utils import (
    get_rolling_windows,
    calculate_simple_average,
    calculate_variance,
    calculate_trend,
    find_closest_rank_games,
    calculate_weighted_average,
    assess_data_quality,
)

__all__ = [
    'OUModel',
    'LGBModel',
    'CatModel',
    'Ensemble3Model',
    'build_ou_features',
    'calculate_rest_features',
    'calculate_venue_features',
    'calculate_conference_features',
    'calculate_game_time_features',
    'get_rolling_windows',
    'calculate_simple_average',
    'calculate_variance',
    'calculate_trend',
    'find_closest_rank_games',
    'calculate_weighted_average',
    'assess_data_quality',
]
