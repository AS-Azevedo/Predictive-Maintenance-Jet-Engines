"""
This is the main script for the training workflow.
It performs the following steps:
1. Loads and processes the data using the data_processing module.
2. Prepares the final datasets for training and testing.
3. Initializes and trains a RandomForestRegressor model.
4. Evaluates the model's performance on the test set using RMSE.
5. Saves the trained model to a file for future use.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

from config import FEATURES, MODEL_PATH
from data_processing import get_processed_data

def train_and_evaluate():
    """
    Orchestrates the model training and evaluation process.
    """
    # 1. Load and process data
    train_df, test_df, rul_df = get_processed_data()
    
    # 2. Prepare final datasets for scikit-learn
    X_train = train_df[FEATURES]
    y_train = train_df['RUL']
    
    # For the test set, the goal is to predict the RUL based on the last available cycle for each engine.
    X_test = test_df.groupby('unit_number').last()[FEATURES]
    y_test = rul_df['RUL']
    
    # 3. Train the model
    print("\nInitializing and training the RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    # 4. Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\nModel performance on the test set:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
    
    # 5. Save the trained model
    print(f"\nSaving the trained model to: {MODEL_PATH}")
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Serialize and save the model object
    joblib.dump(model, MODEL_PATH)
    print("Model saved successfully.")
    
if __name__ == '__main__':
    # This block ensures the training process runs only when the script is executed directly.
    train_and_evaluate()