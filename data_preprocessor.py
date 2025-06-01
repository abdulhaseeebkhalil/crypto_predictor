#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Constants
DATA_DIR = "data"
HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, "BTC-USD_1d.csv")
PROCESSED_DATA_DIR = "processed_data"
SCALER_FILE = os.path.join(PROCESSED_DATA_DIR, "scaler.joblib")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "BTC-USD_1d_processed.csv")
TARGET_COLUMN = "Close" # The column we want to predict

def preprocess_data(input_filepath, output_dir, scaler_filepath, processed_filepath):
    """Loads, cleans, scales, and saves the cryptocurrency data."""
    print(f"Loading data from {input_filepath}...")
    try:
        df = pd.read_csv(input_filepath, index_col="Timestamp", parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Input data file not found at {input_filepath}")
        return None, None

    print(f"Initial data shape: {df.shape}")

    # --- Data Cleaning ---
    # Check for missing values (should have been handled by data_collector, but double-check)
    initial_nan_count = df.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"Warning: Found {initial_nan_count} missing values. Dropping rows with NaNs.")
        df.dropna(inplace=True)
        print(f"Data shape after dropping NaNs: {df.shape}")
    else:
        print("No missing values found.")

    # Ensure target column exists
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column 	'{TARGET_COLUMN}	' not found in the data.")
        return None, None

    # Select the target column for scaling and modeling (can be expanded later)
    data_to_scale = df[[TARGET_COLUMN]]

    # --- Feature Scaling ---
    print(f"Scaling the 	'{TARGET_COLUMN}	' column using MinMaxScaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)

    # Create processed data directory if it doesn	 exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the scaler
    joblib.dump(scaler, scaler_filepath)
    print(f"Scaler saved to {scaler_filepath}")

    # Create a DataFrame for the scaled data
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{TARGET_COLUMN}_scaled"], index=df.index)

    # Save the processed data
    scaled_df.to_csv(processed_filepath)
    print(f"Processed (scaled) data saved to {processed_filepath}")

    return scaled_df, scaler

if __name__ == "__main__":
    processed_df, data_scaler = preprocess_data(
        HISTORICAL_DATA_FILE,
        PROCESSED_DATA_DIR,
        SCALER_FILE,
        PROCESSED_DATA_FILE
    )

    if processed_df is not None:
        print("\nData preprocessing completed successfully.")
        print(f"Processed data head:\n{processed_df.head()}")
    else:
        print("\nData preprocessing failed.")

