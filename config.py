"""Configuration settings for strikeout prediction model."""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
UTILS_DIR = PROJECT_ROOT / "utils"

# Data directories
LINEUPS_DIR = DATA_DIR / "lineups"
RESULTS_DIR = DATA_DIR / "results"
MODEL_HISTORY_DIR = DATA_DIR / "model_history"

# Create directories if they don't exist
for dir_path in [LINEUPS_DIR, RESULTS_DIR, MODEL_HISTORY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    "random_state": 42,
    "test_size": 0.2,
    "val_size": 0.1,
}

# XGBoost parameters
XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "random_state": 42,
}

# LightGBM parameters
LGB_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "random_state": 42,
}

# Feature configuration
FEATURE_CONFIG = {
    "pitcher_stats": [
        "k_per_9",
        "era",
        "whip",
        "avg_velocity",
        "spin_rate",
        "inning",
        "games_started",
    ],
    "handedness_features": [
        "pitcher_handedness",
        "batter_handedness",
        "matchup_type",  # L/L, L/R, R/L, R/R
    ],
    "historical_features": [
        "pitcher_vs_batter_strikeout_pct",
        "pitcher_last_10_k9",
        "pitcher_home_away",
        "team_strikeout_rate",
    ],
}

# Strikeout thresholds for predictions
KO_THRESHOLDS = {
    "conservative": 0.65,  # High confidence
    "moderate": 0.55,      # Medium confidence
    "aggressive": 0.45,    # Lower confidence
}

# Performance tracking
PERFORMANCE_LOG = MODEL_HISTORY_DIR / "performance_log.csv"
MODEL_CHECKPOINTS = MODEL_HISTORY_DIR / "checkpoints"
MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# Prediction output
PREDICTION_OUTPUT = DATA_DIR / "predictions"
PREDICTION_OUTPUT.mkdir(parents=True, exist_ok=True)
