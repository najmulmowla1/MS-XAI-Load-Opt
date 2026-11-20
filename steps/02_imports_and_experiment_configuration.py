# Step 02: Imports and experiment configuration

import os, re, gc, sys, json, math, time, pickle, hashlib, platform, warnings
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

from scipy.stats import norm, wilcoxon
from statsmodels.stats.multitest import multipletests

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import holidays
import optuna
import psutil
import joblib

# -------------------------
# Reproducibility
# -------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# TensorFlow is imported lazily for deep-learning benchmark execution.

# -------------------------
# Input paths
# -------------------------
DATA_DIR = Path(os.environ.get(
    'MS_XAI_DATA_DIR',
    '/content' if Path('/content').exists() else str(Path.cwd() / 'data'),
)).expanduser()
HOURLY_FILE = DATA_DIR / 'hourly_energy_data.csv'
OWID_FILE = DATA_DIR / 'owid_energy_data.csv'

# -------------------------
# Output paths
# -------------------------
OUTPUT_DIR = Path(os.environ.get(
    'MS_XAI_OUTPUT_DIR',
    '/content/ms_xai_load_opt_results' if Path('/content').exists() else str(Path.cwd() / 'results'),
)).expanduser()
FIG_DIR = OUTPUT_DIR / 'figures'
MODEL_DIR = OUTPUT_DIR / 'models'
CACHE_DIR = OUTPUT_DIR / 'cache'
for p in [OUTPUT_DIR, FIG_DIR, MODEL_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

MASTER_EXCEL_PATH = OUTPUT_DIR / 'MS_XAI_Load_Opt_Evidence.xlsx'
SUMMARY_JSON_PATH = OUTPUT_DIR / 'analysis_summary.json'
CONFIG_JSON_PATH = OUTPUT_DIR / 'run_configuration.json'

# -------------------------
# Forecasting definition
# -------------------------
TARGET = 'consumption_MWh'
FORECAST_HORIZON_HOURS = 1
TIMESTAMP_ROLE = 'target_delivery_hour'
ASSUMED_TIMEZONE = 'Europe/Istanbul'  # source timestamps are treated as local wall-clock time

# -------------------------
# Temporal evaluation
# -------------------------
TEST_YEAR = 2023
PRETEST_END_YEAR = 2022
CV_VALIDATION_YEARS = [2020, 2021, 2022]  # expanding-window folds

# -------------------------
# Feature design
# -------------------------
TARGET_LAGS = [1, 2, 3, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 12, 24, 48, 168]
OPERATIONAL_LAGS = [1, 24, 168]
CONSERVATIVE_OPERATIONAL_LAGS = [24, 168]  # latency sensitivity excludes lag-1 generation/price
OWID_LAG_MAIN = 1
OWID_LAG_SENSITIVITY = 2

# Main feature scenario retains the original lag-1 operational variables but explicitly audits the assumption.
MAIN_AVAILABILITY_SCENARIO = 'lag1_operational_assumed_available_at_forecast_origin'
RUN_CONSERVATIVE_LATENCY_SENSITIVITY = True
RUN_OWID_PUBLICATION_LAG_SENSITIVITY = True

# -------------------------
# Model / optimization settings
# -------------------------
N_ESTIMATORS_DEFAULT = 500
N_ESTIMATORS_OPT = 500
RUN_GRID_SEARCH = True
RUN_RANDOM_SEARCH = True
RUN_OPTUNA_XGB = True
RUN_OPTUNA_LGBM = True
RUN_DIFFERENTIAL_EVOLUTION = True

GRID_COMBINATIONS = 24
RANDOM_SEARCH_TRIALS = 24
OPTUNA_XGB_TRIALS = 30
OPTUNA_LGBM_TRIALS = 30
DE_POPULATION_SIZE = 8
DE_ITERATIONS = 8

RUN_PROPHET = True
RUN_DEEP_LEARNING = True
DL_WINDOW = 168
DL_MAX_EPOCHS = 50
DL_PATIENCE = 6
DL_BATCH_SIZE = 256
DL_MODELS = ['LSTM', 'GRU', 'CNN_LSTM', 'Transformer', 'CNN_Transformer']

# -------------------------
# Statistical / uncertainty settings
# -------------------------
HAC_LAGS_PRIMARY = 24
HAC_LAGS_SENSITIVITY = [24, 168]
BOOTSTRAP_BLOCK_HOURS = 24
BOOTSTRAP_REPS = 2000
BOOTSTRAP_CI = 0.95
CONFORMAL_NOMINAL_LEVELS = [0.80, 0.90, 0.95]
CONFORMAL_BLOCK_SENSITIVITY = [24, 168]
SHAP_SAMPLE_SIZE = 3000

# -------------------------
# Runtime controls
# -------------------------
SAVE_INTERMEDIATE_CSV = True
VERBOSE_MODEL_FITS = False

RUN_CONFIG = {
    'random_state': RANDOM_STATE,
    'target': TARGET,
    'forecast_horizon_hours': FORECAST_HORIZON_HOURS,
    'timestamp_role': TIMESTAMP_ROLE,
    'assumed_timezone': ASSUMED_TIMEZONE,
    'test_year': TEST_YEAR,
    'pretest_end_year': PRETEST_END_YEAR,
    'cv_validation_years': CV_VALIDATION_YEARS,
    'target_lags': TARGET_LAGS,
    'rolling_windows': ROLL_WINDOWS,
    'operational_lags': OPERATIONAL_LAGS,
    'conservative_operational_lags': CONSERVATIVE_OPERATIONAL_LAGS,
    'main_availability_scenario': MAIN_AVAILABILITY_SCENARIO,
    'grid_combinations': GRID_COMBINATIONS,
    'random_search_trials': RANDOM_SEARCH_TRIALS,
    'optuna_xgb_trials': OPTUNA_XGB_TRIALS,
    'optuna_lgbm_trials': OPTUNA_LGBM_TRIALS,
    'de_population_size': DE_POPULATION_SIZE,
    'de_iterations': DE_ITERATIONS,
    'run_prophet': RUN_PROPHET,
    'run_deep_learning': RUN_DEEP_LEARNING,
    'dl_window': DL_WINDOW,
    'dl_max_epochs': DL_MAX_EPOCHS,
    'dl_patience': DL_PATIENCE,
    'hac_lags_primary': HAC_LAGS_PRIMARY,
    'bootstrap_block_hours': BOOTSTRAP_BLOCK_HOURS,
    'bootstrap_reps': BOOTSTRAP_REPS,
    'conformal_nominal_levels': CONFORMAL_NOMINAL_LEVELS,
}
with open(CONFIG_JSON_PATH, 'w') as f:
    json.dump(RUN_CONFIG, f, indent=2)

print('Experiment configuration initialized')
print('Output:', OUTPUT_DIR)
