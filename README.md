# Cryptocurrency Price Prediction AI Agent

## Overview
This application uses LSTM neural networks to predict cryptocurrency prices based on historical data. It provides a user-friendly interface for selecting cryptocurrencies, viewing historical trends, and getting price predictions.

## Features
- Historical cryptocurrency data collection using Yahoo Finance API
- Real-time price fetching
- LSTM-based neural network for time-series prediction
- Interactive visualizations of historical prices and predictions
- Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- Volatility analysis
- Price movement alerts
- Downloadable analysis reports

## Installation

### Requirements
- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Usage
1. Select a cryptocurrency from the dropdown menu or enter a custom symbol
2. Choose the historical data range and interval
3. View historical price data and statistics
4. Generate price predictions
5. Analyze technical indicators
6. Download comprehensive reports

## Project Structure
- `app.py`: Main Streamlit application
- `data_collector.py`: Functions for collecting historical and real-time data
- `data_preprocessor.py`: Data cleaning and preprocessing
- `time_series_formatter.py`: Prepares data in time-series format for LSTM
- `model_builder.py`: LSTM model architecture definition
- `model_trainer.py`: Model training and validation
- `predictor.py`: Real-time prediction functionality
- `model_evaluator.py`: Model performance evaluation

## Model Performance
The LSTM model has been evaluated on historical cryptocurrency data with the following metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
- Directional Accuracy

Detailed evaluation results can be found in the `evaluation` directory.

## Disclaimer
This tool is for educational purposes only. Cryptocurrency investments are subject to high market risk. Past performance is not indicative of future results.
