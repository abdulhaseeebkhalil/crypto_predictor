#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os

# Constants
PROCESSED_DATA_DIR = "processed_data"
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "BTC-USD_1d_processed.csv")
SEQUENCE_LENGTH = 60  # Look-back period (e.g., use last 60 days to predict the next day)
TIMESERIES_DATA_DIR = "timeseries_data"
X_FILE = os.path.join(TIMESERIES_DATA_DIR, "X_data.npy")
Y_FILE = os.path.join(TIMESERIES_DATA_DIR, "y_data.npy")

def create_time_series_data(input_filepath, sequence_length, output_dir, x_filepath, y_filepath):
    """Loads processed data and creates time series sequences for LSTM."""
    print(f"Loading processed data from {input_filepath}...")
    try:
        df = pd.read_csv(input_filepath, index_col="Timestamp", parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {input_filepath}")
        return None, None

    if df.empty:
        print("Error: Processed data file is empty.")
        return None, None

    # Assuming the scaled data is in the first column
    scaled_data = df.iloc[:, 0].values
    print(f"Using data column: {df.columns[0]}")
    print(f"Total data points: {len(scaled_data)}")

    if len(scaled_data) <= sequence_length:
        print(f"Error: Not enough data ({len(scaled_data)} points) to create sequences of length {sequence_length}.")
        return None, None

    # --- Create Sequences ---
    X = []
    y = []
    print(f"Creating sequences with length {sequence_length}...")
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i]) # Sequence of past data
        y.append(scaled_data[i])                   # Next data point (target)

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"Shape of X before reshape: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # Reshape X for LSTM input [samples, time_steps, features]
    # In this case, features = 1 (just the scaled price)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(f"Shape of X after reshape (samples, time_steps, features): {X.shape}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the arrays
    np.save(x_filepath, X)
    np.save(y_filepath, y)
    print(f"Time series data saved: X -> {x_filepath}, y -> {y_filepath}")

    return X, y

if __name__ == "__main__":
    X_data, y_data = create_time_series_data(
        PROCESSED_DATA_FILE,
        SEQUENCE_LENGTH,
        TIMESERIES_DATA_DIR,
        X_FILE,
        Y_FILE
    )

    if X_data is not None and y_data is not None:
        print("\nTime series data preparation completed successfully.")
    else:
        print("\nTime series data preparation failed.")

