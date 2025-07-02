"""
This module handles all data loading and feature engineering tasks.
Its main purpose is to take the raw text files and transform them into
clean, feature-rich DataFrames ready for model training.
"""

import pandas as pd
from config import TRAIN_FILE, TEST_FILE, RUL_FILE, ALL_COLS, SENSOR_COLS, WINDOW_SIZE, LAG_PERIOD

def load_data():
    """
    Loads the training, test, and RUL ground truth data from their respective files.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
               (train_df, test_df, rul_df).
    """
    train_df = pd.read_csv(TRAIN_FILE, sep=r'\s+', header=None, names=ALL_COLS)
    test_df = pd.read_csv(TEST_FILE, sep=r'\s+', header=None, names=ALL_COLS)
    rul_df = pd.read_csv(RUL_FILE, sep=r'\s+', header=None, names=['RUL'])
    return train_df, test_df, rul_df

def engineer_features(df):
    """
    Applies all feature engineering steps to a given dataframe.

    This includes calculating the RUL for training data and creating various
    time-series features like rolling statistics, derivatives, and lags.

    Args:
        df (pd.DataFrame): The input dataframe (either train or test).

    Returns:
        pd.DataFrame: The dataframe with all engineered features.
    """
    # If it's the training set, calculate the RUL (target variable).
    # The RUL is the number of cycles remaining until failure.
    if 'RUL' not in df.columns:
        max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycles']
        df = pd.merge(df, max_cycles, on='unit_number', how='left')
        df['RUL'] = df['max_cycles'] - df['time_in_cycles']
        df.drop(columns=['max_cycles'], inplace=True)

    # --- Feature Engineering ---
    # Group by engine unit to ensure calculations are done per engine.
    grouped = df.groupby('unit_number')

    # Rolling statistics to smooth out sensor noise and capture trends.
    for col in SENSOR_COLS:
        df[f'{col}_mean'] = grouped[col].transform(lambda x: x.rolling(WINDOW_SIZE, 1).mean())
        df[f'{col}_std'] = grouped[col].transform(lambda x: x.rolling(WINDOW_SIZE, 1).std())
    
    # Rate of change (derivative) to capture how fast sensor values are changing.
    for col in SENSOR_COLS:
        df[f'{col}_deriv'] = grouped[col].diff()
        
    # Lag features to give the model a "memory" of past sensor values.
    for col in SENSOR_COLS:
        df[f'{col}_lag{LAG_PERIOD}'] = grouped[col].shift(LAG_PERIOD)
        
    # After creating derivative and lag features, there will be NaN values
    # at the beginning of each engine's lifecycle. We fill them with 0.
    df.fillna(0, inplace=True)
    
    return df

def get_processed_data():
    """
    Main function to orchestrate the data loading and processing workflow.

    Returns:
        tuple: A tuple containing three processed pandas DataFrames:
               (train_processed, test_processed, rul_df).
    """
    train_df, test_df, rul_df = load_data()
    
    print("Processing training data...")
    train_processed = engineer_features(train_df.copy())
    
    print("Processing test data...")
    test_processed = engineer_features(test_df.copy())
    
    return train_processed, test_processed, rul_df