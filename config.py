"""
This file centralizes all configurations for the project.
By changing the values here, you can easily adapt the project to different datasets
or experiment with different model parameters without changing the core logic.
"""

# --- File Paths ---
# Paths to the raw data files.
TRAIN_FILE = 'data/train_FD001.txt'
TEST_FILE = 'data/test_FD001.txt'
RUL_FILE = 'data/RUL_FD001.txt'

# Path to save the trained model artifact.
MODEL_PATH = 'saved_models/rul_model.pkl'


# --- Data Column Names ---
# The base column names are defined as per the dataset's documentation.
BASE_COLS = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
SENSOR_COLS = [f'sensor_meas_{i}' for i in range(1, 22)]
ALL_COLS = BASE_COLS + SENSOR_COLS


# --- Feature Engineering Parameters ---
# Window size for calculating rolling statistics (mean, std).
WINDOW_SIZE = 10
# The number of past cycles to look back for lag features.
LAG_PERIOD = 5


# --- Final Feature List for the Model ---
# This list defines exactly which columns will be used as features for training the model.
# Centralizing it here makes it easy to add or remove features for experimentation.
FEATURES = []
FEATURES.extend(SENSOR_COLS)
FEATURES.extend([f'{col}_mean' for col in SENSOR_COLS])
FEATURES.extend([f'{col}_std' for col in SENSOR_COLS])
FEATURES.extend([f'{col}_deriv' for col in SENSOR_COLS])
FEATURES.extend([f'{col}_lag{LAG_PERIOD}' for col in SENSOR_COLS])