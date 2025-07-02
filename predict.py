"""
This script provides an example of how to load the pre-trained model
and use it to make a prediction on new, unseen data.
"""

import pandas as pd
import joblib
from config import MODEL_PATH, FEATURES

def make_prediction(input_data):
    """
    Loads the trained model and makes a RUL prediction on new data.

    Args:
        input_data (pd.DataFrame): A DataFrame containing a single row of sensor data
                                   with the same feature columns as the training set.

    Returns:
        np.array: An array containing the single predicted RUL value.
    """
    try:
        # Load the serialized model from the file
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please run 'python train.py' first to train and save the model.")
        return None
    
    # Ensure the input data has the same columns in the same order as the model was trained on.
    input_data = input_data[FEATURES]
    
    # Use the model to predict the RUL
    prediction = model.predict(input_data)
    
    return prediction

if __name__ == '__main__':
    # This block demonstrates how to use the make_prediction function.
    
    print("Creating a sample data point for prediction...")

    # 1. Create a dictionary for a single data point (one engine at one point in time).
    #    Initialize all feature values to 0.
    sample_data = {feature: [0] for feature in FEATURES}

    # 2. Update some key feature values to simulate a real sensor reading.
    #    In a real application, this data would come from an active sensor.
    sample_data['sensor_meas_2'] = [642.5]
    sample_data['sensor_meas_7'] = [15.2]
    sample_data['sensor_meas_11'] = [47.2]
    sample_data['sensor_meas_2_mean'] = [642.3]
    sample_data['sensor_meas_2_std'] = [0.5]
    sample_data['sensor_meas_7_deriv'] = [-0.01]
    sample_data['sensor_meas_11_lag5'] = [47.0]

    # 3. Convert the dictionary into a pandas DataFrame.
    #    All values are lists of length 1, so this creates a DataFrame with one row.
    sample_df = pd.DataFrame(sample_data)
    
    # 4. Make the prediction
    predicted_rul = make_prediction(sample_df)
    
    # 5. Print the result
    if predicted_rul is not None:
        print("\n--- ✅ Prediction Example ---")
        print(f"Predicted RUL for the sample data: {predicted_rul[0]:.2f} cycles")