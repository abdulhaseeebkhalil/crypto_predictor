#!/usr/bin/env python3

import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import the model building function
from model_builder import build_lstm_model, save_model

# Constants
TIMESERIES_DATA_DIR = "timeseries_data"
X_FILE = os.path.join(TIMESERIES_DATA_DIR, "X_data.npy")
Y_FILE = os.path.join(TIMESERIES_DATA_DIR, "y_data.npy")
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "crypto_predictor_lstm.keras")
HISTORY_FILE = os.path.join(MODEL_DIR, "training_history.npy")

# Training Hyperparameters
TEST_SIZE = 0.2 # Proportion of data to use for testing
EPOCHS = 50     # Number of training epochs (can be adjusted)
BATCH_SIZE = 32 # Batch size for training
EARLY_STOPPING_PATIENCE = 10 # Patience for early stopping

def train_and_validate_model(x_filepath, y_filepath, model_filepath, history_filepath):
    """Loads data, splits it, trains the model, validates, and saves results."""
    print("--- Starting Model Training and Validation ---")

    # --- Load Data ---
    print(f"Loading time series data from {x_filepath} and {y_filepath}...")
    try:
        X = np.load(x_filepath)
        y = np.load(y_filepath)
    except FileNotFoundError:
        print(f"Error: Data files not found. Please run time_series_formatter.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    # --- Split Data ---
    print(f"Splitting data into training and testing sets (Test size: {TEST_SIZE * 100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, shuffle=False) # Don't shuffle time series data
    print(f"Training set shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shapes: X_test={X_test.shape}, y_test={y_test.shape}")

    # --- Build Model ---
    # Assumes SEQUENCE_LENGTH and FEATURES are correctly set in model_builder.py
    model = build_lstm_model()
    if model is None:
        print("Error: Failed to build the model.")
        return None

    # --- Configure Callbacks ---
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor=	'val_loss'	, patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    # Model checkpoint to save the best model during training
    model_checkpoint = ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor=	'val_loss'	, mode=	'min'	)

    print(f"\n--- Training Model (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}) ---")
    # --- Train Model ---
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1, # Use 10% of training data for validation during training
        callbacks=[early_stopping, model_checkpoint],
        verbose=1 # Show progress
    )

    print("\n--- Evaluating Model on Test Set ---")
    # --- Validate Model ---
    # Load the best model saved by ModelCheckpoint
    try:
        best_model = tf.keras.models.load_model(model_filepath)
        test_loss = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (Mean Squared Error): {test_loss}")
    except Exception as e:
        print(f"Error loading or evaluating the best model: {e}")
        print("Evaluating the model as it is after the last epoch.")
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (Mean Squared Error) - Last Epoch Model: {test_loss}")
        # Save the last epoch model if the best couldn't be loaded/evaluated
        save_model(model, model_filepath.replace(".keras", "_last.keras"))


    # --- Save Training History ---
    try:
        np.save(history_filepath, history.history)
        print(f"Training history saved to {history_filepath}")
    except Exception as e:
        print(f"Error saving training history: {e}")

    print("\n--- Model Training and Validation Finished ---")
    return history

if __name__ == "__main__":
    training_history = train_and_validate_model(
        X_FILE,
        Y_FILE,
        MODEL_FILE,
        HISTORY_FILE
    )

    if training_history is not None:
        print("\nModel training script completed successfully.")
    else:
        print("\nModel training script encountered errors.")

