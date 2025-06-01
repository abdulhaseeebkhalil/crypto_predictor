#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

# Import functions from other scripts (ensure they are accessible)
# Assuming data_collector.py is in the same directory or path is configured
from data_collector import get_historical_data_yahoo, get_real_time_price_yahoo

# Constants
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "crypto_predictor_lstm.keras")
PROCESSED_DATA_DIR = "processed_data"
SCALER_FILE = os.path.join(PROCESSED_DATA_DIR, "scaler.joblib")
SEQUENCE_LENGTH = 60 # Must match the sequence length used during training
TARGET_COLUMN = "Close"

def load_model_and_scaler(model_filepath, scaler_filepath):
    """Loads the trained Keras model and the scaler."""
    print("Loading model and scaler...")
    try:
        model = tf.keras.models.load_model(model_filepath)
        print(f"Model loaded successfully from {model_filepath}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    try:
        scaler = joblib.load(scaler_filepath)
        print(f"Scaler loaded successfully from {scaler_filepath}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_filepath}")
        return model, None # Return model even if scaler fails, maybe handle later
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return model, None

    return model, scaler

def prepare_prediction_input(symbol, sequence_length, scaler):
    """Fetches recent data, scales it, and prepares the input sequence for prediction."""
    print(f"\nPreparing input data for {symbol} prediction...")
    # Fetch slightly more data than needed to ensure we have enough points
    # Yahoo range needs careful handling; fetch ~90 days to be safe for a 60-day sequence
    range_to_fetch = "90d" # Fetch last 90 days
    interval = "1d"

    # Fetch historical data
    # Note: Reusing the function from data_collector
    recent_data_df = get_historical_data_yahoo(symbol, interval, range_to_fetch)

    if recent_data_df is None or recent_data_df.empty:
        print("Error: Could not fetch recent historical data for prediction input.")
        return None

    # Ensure we have enough data points
    if len(recent_data_df) < sequence_length:
        print(f"Error: Not enough historical data ({len(recent_data_df)} points) to form a sequence of length {sequence_length}.")
        return None

    # Select the target column and take the last `sequence_length` points
    try:
        # Ensure the column exists and handle potential KeyError
        if TARGET_COLUMN not in recent_data_df.columns:
             print(f"Error: Target column 		'{TARGET_COLUMN}		' not found in fetched data.")
             # Try 'Adj Close' as a fallback if it exists
             if 'Adj Close' in recent_data_df.columns:
                 print("Using 'Adj Close' as fallback.")
                 input_data = recent_data_df[		'Adj Close		'].values[-sequence_length:]
             else:
                 print("Cannot find suitable column for prediction input.")
                 return None
        else:
             input_data = recent_data_df[TARGET_COLUMN].values[-sequence_length:]

        # Check for NaNs in the final input sequence
        if np.isnan(input_data).any():
            print("Error: NaN values found in the input sequence data. Cannot proceed.")
            return None

        # Scale the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data.reshape(-1, 1))

        # Reshape for LSTM input [1, sequence_length, features]
        X_pred = np.reshape(input_data_scaled, (1, sequence_length, 1))
        print(f"Input sequence prepared with shape: {X_pred.shape}")
        return X_pred

    except KeyError as e:
        print(f"Error accessing column for prediction: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during input preparation: {e}")
        return None

def make_prediction(model, X_input, scaler):
    """Makes a prediction using the model and inverse transforms the result."""
    if model is None or X_input is None or scaler is None:
        print("Error: Model, input data, or scaler is missing for prediction.")
        return None

    print("\nMaking prediction...")
    try:
        predicted_scaled = model.predict(X_input)
        print(f"Raw scaled prediction: {predicted_scaled[0][0]}")

        # Inverse transform the prediction to get the actual price
        predicted_price = scaler.inverse_transform(predicted_scaled)
        print(f"Predicted price (inverse transformed): {predicted_price[0][0]}")
        return predicted_price[0][0]
    except Exception as e:
        print(f"Error during prediction or inverse transform: {e}")
        return None

if __name__ == "__main__":
    symbol_to_predict = "BTC-USD" # Example symbol

    # 1. Load Model and Scaler
    loaded_model, loaded_scaler = load_model_and_scaler(MODEL_FILE, SCALER_FILE)

    if loaded_model and loaded_scaler:
        # 2. Prepare Input Data
        prediction_input = prepare_prediction_input(symbol_to_predict, SEQUENCE_LENGTH, loaded_scaler)

        if prediction_input is not None:
            # 3. Make Prediction
            predicted_value = make_prediction(loaded_model, prediction_input, loaded_scaler)

            if predicted_value is not None:
                print(f"\n--- Prediction Result for {symbol_to_predict} ---")
                print(f"Predicted next day		's Close price: {predicted_value:.2f}")

                # Optional: Fetch current price for context
                current_price = get_real_time_price_yahoo(symbol_to_predict)
                if current_price:
                    print(f"Current market price: {current_price:.2f}")
            else:
                print("\nPrediction failed.")
        else:
            print("\nFailed to prepare input for prediction.")
    else:
        print("\nFailed to load model or scaler. Cannot proceed with prediction.")

