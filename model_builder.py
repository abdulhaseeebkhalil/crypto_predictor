#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Constants
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "crypto_predictor_lstm.keras") # Using .keras format
SEQUENCE_LENGTH = 60 # Must match the sequence length used in time_series_formatter.py
FEATURES = 1 # Number of features per time step (just scaled price)

# --- Hyperparameter Configuration ---
# These are initial settings and can be tuned during the training/validation phase.
LSTM_UNITS_1 = 50       # Number of units in the first LSTM layer
LSTM_UNITS_2 = 50       # Number of units in the second LSTM layer
DROPOUT_RATE = 0.2      # Dropout rate for regularization to prevent overfitting
DENSE_UNITS = 25        # Number of units in the intermediate Dense layer
OPTIMIZER = 	"adam"				# Optimizer algorithm (Adam is a common choice)
LOSS_FUNCTION = "mean_squared_error" # Loss function for regression tasks
# Training specific hyperparameters (epochs, batch_size) will be set in the training script.

def build_lstm_model(sequence_length=SEQUENCE_LENGTH, features=FEATURES):
    """Builds and compiles the LSTM model architecture using configured hyperparameters."""
    print("Building LSTM model with the following configuration:")
    print(f"  - LSTM Layer 1 Units: {LSTM_UNITS_1}")
    print(f"  - LSTM Layer 2 Units: {LSTM_UNITS_2}")
    print(f"  - Dropout Rate: {DROPOUT_RATE}")
    print(f"  - Dense Layer Units: {DENSE_UNITS}")
    print(f"  - Optimizer: {OPTIMIZER}")
    print(f"  - Loss Function: {LOSS_FUNCTION}")

    model = Sequential()

    # --- Model Architecture ---
    # Layer 1: LSTM layer with return_sequences=True for stacking LSTM layers
    model.add(LSTM(units=LSTM_UNITS_1, return_sequences=True, input_shape=(sequence_length, features)))
    model.add(Dropout(DROPOUT_RATE))

    # Layer 2: Another LSTM layer
    model.add(LSTM(units=LSTM_UNITS_2, return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))

    # Layer 3: Dense layer for processing LSTM output
    model.add(Dense(units=DENSE_UNITS))

    # Layer 4: Output layer - Dense layer with 1 unit for predicting the next value
    model.add(Dense(units=1))

    # --- Compilation ---
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)

    print("\nModel built successfully.")
    model.summary() # Print model summary

    return model

def save_model(model, model_filepath):
    """Saves the Keras model to the specified path."""
    model_dir = os.path.dirname(model_filepath)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    try:
        model.save(model_filepath)
        print(f"Model saved to {model_filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    lstm_model = build_lstm_model()

    # Note: We build the model here, but training happens in a separate script.
    # Saving the untrained model structure is optional but can be useful.
    print("\nModel building script finished.")
    # Uncomment to save the initial model structure:
    # save_model(lstm_model, MODEL_FILE)

