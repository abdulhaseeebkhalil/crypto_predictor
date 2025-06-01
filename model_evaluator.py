#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# Import functions from other scripts
from data_collector import get_historical_data_yahoo
from predictor import load_model_and_scaler

# Constants
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "crypto_predictor_lstm.keras")
PROCESSED_DATA_DIR = "processed_data"
SCALER_FILE = os.path.join(PROCESSED_DATA_DIR, "scaler.joblib")
EVALUATION_DIR = "evaluation"
EVALUATION_RESULTS_FILE = os.path.join(EVALUATION_DIR, "model_evaluation_results.csv")
EVALUATION_PLOTS_DIR = os.path.join(EVALUATION_DIR, "plots")
SEQUENCE_LENGTH = 60

def evaluate_model(symbol="BTC-USD", test_period="6mo", interval="1d"):
    """
    Evaluates the model performance by comparing predictions with actual prices
    for a specified test period.
    """
    print(f"Evaluating model performance for {symbol} over {test_period} with {interval} interval...")
    
    # Create evaluation directory if it doesn't exist
    if not os.path.exists(EVALUATION_DIR):
        os.makedirs(EVALUATION_DIR)
        print(f"Created directory: {EVALUATION_DIR}")
    
    if not os.path.exists(EVALUATION_PLOTS_DIR):
        os.makedirs(EVALUATION_PLOTS_DIR)
        print(f"Created directory: {EVALUATION_PLOTS_DIR}")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(MODEL_FILE, SCALER_FILE)
    if model is None or scaler is None:
        print("Failed to load model or scaler. Cannot proceed with evaluation.")
        return None
    
    # Fetch test data
    test_data = get_historical_data_yahoo(symbol, interval, test_period)
    if test_data is None or test_data.empty:
        print(f"Failed to fetch test data for {symbol}.")
        return None
    
    print(f"Fetched test data with {len(test_data)} data points.")
    
    # Ensure we have enough data for evaluation
    if len(test_data) <= SEQUENCE_LENGTH:
        print(f"Not enough data points ({len(test_data)}) for evaluation. Need at least {SEQUENCE_LENGTH + 1} points.")
        return None
    
    # Prepare data for evaluation
    close_prices = test_data['Close'].values
    
    # Scale the data
    scaled_prices = scaler.transform(close_prices.reshape(-1, 1)).flatten()
    
    # Create sequences for prediction
    X = []
    y_true = []
    
    for i in range(SEQUENCE_LENGTH, len(scaled_prices)):
        X.append(scaled_prices[i-SEQUENCE_LENGTH:i])
        y_true.append(scaled_prices[i])
    
    X = np.array(X)
    y_true = np.array(y_true)
    
    # Reshape X for LSTM input [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    print(f"Prepared {len(X)} sequences for evaluation.")
    
    # Make predictions
    y_pred_scaled = model.predict(X)
    
    # Inverse transform predictions and actual values
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred))
    r2 = r2_score(y_true_original, y_pred)
    mape = np.mean(np.abs((y_true_original - y_pred) / y_true_original)) * 100
    
    # Calculate directional accuracy (how often the model correctly predicts the direction of price movement)
    actual_direction = np.diff(y_true_original) > 0
    predicted_direction = np.diff(y_pred) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # Save evaluation results
    results = {
        'Symbol': symbol,
        'Test Period': test_period,
        'Interval': interval,
        'Data Points': len(test_data),
        'Sequences': len(X),
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape,
        'Directional Accuracy (%)': directional_accuracy,
        'Evaluation Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(EVALUATION_RESULTS_FILE, index=False)
    print(f"Evaluation results saved to {EVALUATION_RESULTS_FILE}")
    
    # Create evaluation plots
    create_evaluation_plots(test_data.index[-len(y_true_original):], y_true_original, y_pred, symbol)
    
    print("\nEvaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    return results

def create_evaluation_plots(dates, y_true, y_pred, symbol):
    """Creates and saves evaluation plots."""
    # Plot 1: Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual Price', color='blue')
    plt.plot(dates, y_pred, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'Actual vs Predicted Prices - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(EVALUATION_PLOTS_DIR, f"{symbol}_actual_vs_predicted.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # Plot 2: Prediction Error
    plt.figure(figsize=(12, 6))
    error = y_true - y_pred
    plt.plot(dates, error, color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(f'Prediction Error - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Error (Actual - Predicted)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(EVALUATION_PLOTS_DIR, f"{symbol}_prediction_error.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # Plot 3: Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(error, bins=50, alpha=0.75, color='purple')
    plt.title(f'Error Distribution - {symbol}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(EVALUATION_PLOTS_DIR, f"{symbol}_error_distribution.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    # Plot 4: Scatter plot of Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'Actual vs Predicted Scatter Plot - {symbol}')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(EVALUATION_PLOTS_DIR, f"{symbol}_scatter_plot.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Evaluate for Bitcoin with 6 months of test data
    results = evaluate_model("BTC-USD", "6mo", "1d")
    
    if results:
        print("\nModel evaluation completed successfully.")
    else:
        print("\nModel evaluation failed.")
