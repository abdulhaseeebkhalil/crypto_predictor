#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import base64
from io import BytesIO

# Import functions from other scripts
from data_collector import get_historical_data_yahoo, get_real_time_price_yahoo
from predictor import load_model_and_scaler, prepare_prediction_input, make_prediction

# Constants
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "crypto_predictor_lstm.keras")
PROCESSED_DATA_DIR = "processed_data"
SCALER_FILE = os.path.join(PROCESSED_DATA_DIR, "scaler.joblib")
DATA_DIR = "data"
SEQUENCE_LENGTH = 60

# Page configuration
st.set_page_config(
    page_title="Cryptocurrency Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .warning-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .alert-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FF9800;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">Cryptocurrency Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
This application uses LSTM neural networks to predict cryptocurrency prices based on historical data.
Select a cryptocurrency symbol, view historical trends, and get price predictions.
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.title("Settings")

# Cryptocurrency selection
crypto_options = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Litecoin": "LTC-USD",
    "Bitcoin Cash": "BCH-USD",
    "Ripple": "XRP-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",
    "Polkadot": "DOT-USD",
    "Solana": "SOL-USD",
    "Custom": "CUSTOM"
}

selected_crypto_name = st.sidebar.selectbox(
    "Select Cryptocurrency",
    list(crypto_options.keys())
)

# Handle custom symbol input
if selected_crypto_name == "Custom":
    custom_symbol = st.sidebar.text_input("Enter Custom Symbol (Yahoo Finance format, e.g., BTC-USD)")
    if custom_symbol:
        selected_symbol = custom_symbol
    else:
        selected_symbol = "BTC-USD"  # Default if no custom symbol is entered
else:
    selected_symbol = crypto_options[selected_crypto_name]

# Time range selection for historical data
time_range_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Max": "max"
}

selected_time_range_name = st.sidebar.selectbox(
    "Historical Data Range",
    list(time_range_options.keys())
)
selected_time_range = time_range_options[selected_time_range_name]

# Interval selection
interval_options = {
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo"
}

selected_interval_name = st.sidebar.selectbox(
    "Data Interval",
    list(interval_options.keys())
)
selected_interval = interval_options[selected_interval_name]

# Advanced settings expander
with st.sidebar.expander("Advanced Settings"):
    alert_threshold = st.slider(
        "Alert Threshold (%)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Set the threshold for price change alerts"
    )
    
    show_technical_indicators = st.checkbox(
        "Show Technical Indicators",
        value=True,
        help="Display technical indicators like Moving Averages and RSI"
    )
    
    prediction_days = st.slider(
        "Prediction Days",
        min_value=1,
        max_value=7,
        value=1,
        step=1,
        help="Number of days to predict ahead (experimental for multi-day)"
    )

# Function to load historical data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_historical_data(symbol, interval, range_str):
    return get_historical_data_yahoo(symbol, interval, range_str)

# Function to get current price
def get_current_price(symbol):
    return get_real_time_price_yahoo(symbol)

# Function to make prediction
def predict_next_price(symbol, days=1):
    # Load model and scaler
    model, scaler = load_model_and_scaler(MODEL_FILE, SCALER_FILE)
    
    if model is None or scaler is None:
        return None, "Failed to load model or scaler."
    
    # Prepare input data
    prediction_input = prepare_prediction_input(symbol, SEQUENCE_LENGTH, scaler)
    
    if prediction_input is None:
        return None, "Insufficient historical data for prediction."
    
    # Make prediction for first day
    predicted_value = make_prediction(model, prediction_input, scaler)
    
    if predicted_value is None:
        return None, "Prediction failed."
    
    # For single day prediction, return the result
    if days <= 1:
        return predicted_value, None
    
    # For multi-day prediction (experimental)
    predictions = [predicted_value]
    current_input = prediction_input.copy()
    
    # Iteratively predict subsequent days
    for _ in range(1, days):
        # Update the input sequence by removing the oldest value and adding the latest prediction
        current_input[0, :-1, 0] = current_input[0, 1:, 0]
        current_input[0, -1, 0] = scaler.transform([[predictions[-1]]])[0][0]
        
        # Make the next prediction
        next_pred = make_prediction(model, current_input, scaler)
        if next_pred is None:
            break
        predictions.append(next_pred)
    
    return predictions, None

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_tech = df.copy()
    
    # Moving Averages
    df_tech['MA20'] = df_tech['Close'].rolling(window=20).mean()
    df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
    df_tech['MA200'] = df_tech['Close'].rolling(window=200).mean()
    
    # Relative Strength Index (RSI)
    delta = df_tech['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    
    rs = gain / loss
    df_tech['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df_tech['MA20_std'] = df_tech['Close'].rolling(window=20).std()
    df_tech['upper_band'] = df_tech['MA20'] + (df_tech['MA20_std'] * 2)
    df_tech['lower_band'] = df_tech['MA20'] - (df_tech['MA20_std'] * 2)
    
    # MACD
    df_tech['EMA12'] = df_tech['Close'].ewm(span=12, adjust=False).mean()
    df_tech['EMA26'] = df_tech['Close'].ewm(span=26, adjust=False).mean()
    df_tech['MACD'] = df_tech['EMA12'] - df_tech['EMA26']
    df_tech['Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
    
    return df_tech

# Function to generate downloadable report
def generate_report(symbol, historical_data, current_price, predicted_price, volatility_data):
    buffer = BytesIO()
    
    # Create a simple HTML report
    html = f"""
    <html>
    <head>
        <title>Cryptocurrency Analysis Report - {symbol}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #1E88E5; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .highlight {{ background-color: #E8F5E9; }}
        </style>
    </head>
    <body>
        <h1>Cryptocurrency Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Overview - {symbol}</h2>
            <p>Current Price: ${current_price:.2f}</p>
    """
    
    # Add prediction information if available
    if isinstance(predicted_price, list):
        html += f"<p>Predicted Prices for Next {len(predicted_price)} Days:</p><ul>"
        for i, price in enumerate(predicted_price):
            html += f"<li>Day {i+1}: ${price:.2f}</li>"
        html += "</ul>"
    elif predicted_price:
        html += f"<p>Predicted Next Price: ${predicted_price:.2f}</p>"
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        direction = "increase" if price_change > 0 else "decrease"
        html += f"<p>Expected {direction} of ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</p>"
    
    # Add historical data summary
    if historical_data is not None and not historical_data.empty:
        html += f"""
        <div class="section">
            <h2>Historical Data Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Highest Price</td>
                    <td>${historical_data['High'].max():.2f}</td>
                </tr>
                <tr>
                    <td>Lowest Price</td>
                    <td>${historical_data['Low'].min():.2f}</td>
                </tr>
                <tr>
                    <td>Average Price</td>
                    <td>${historical_data['Close'].mean():.2f}</td>
                </tr>
                <tr>
                    <td>Price Range</td>
                    <td>${historical_data['High'].max() - historical_data['Low'].min():.2f}</td>
                </tr>
            </table>
        </div>
        """
    
    # Add volatility information
    if 'Volatility' in historical_data.columns:
        recent_volatility = historical_data['Volatility'].iloc[-1]
        avg_volatility = historical_data['Volatility'].mean()
        
        if recent_volatility < avg_volatility * 0.5:
            volatility_message = "Very Low: The market is currently very stable compared to historical patterns."
        elif recent_volatility < avg_volatility * 0.8:
            volatility_message = "Low: The market is showing below-average volatility."
        elif recent_volatility < avg_volatility * 1.2:
            volatility_message = "Normal: The market is showing typical volatility levels."
        elif recent_volatility < avg_volatility * 2:
            volatility_message = "High: The market is more volatile than usual."
        else:
            volatility_message = "Very High: The market is experiencing extreme volatility."
        
        html += f"""
        <div class="section">
            <h2>Volatility Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Current Volatility</td>
                    <td>{recent_volatility:.2f}%</td>
                </tr>
                <tr>
                    <td>Average Volatility</td>
                    <td>{avg_volatility:.2f}%</td>
                </tr>
                <tr>
                    <td>Volatility Assessment</td>
                    <td>{volatility_message}</td>
                </tr>
            </table>
        </div>
        """
    
    # Add recent price data
    if historical_data is not None and not historical_data.empty:
        html += """
        <div class="section">
            <h2>Recent Price Data</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Open</th>
                    <th>High</th>
                    <th>Low</th>
                    <th>Close</th>
                    <th>Volume</th>
                </tr>
        """
        
        for idx, row in historical_data.tail(10).iterrows():
            html += f"""
                <tr>
                    <td>{idx.strftime('%Y-%m-%d')}</td>
                    <td>${row['Open']:.2f}</td>
                    <td>${row['High']:.2f}</td>
                    <td>${row['Low']:.2f}</td>
                    <td>${row['Close']:.2f}</td>
                    <td>{int(row['Volume']):,}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Add disclaimer
    html += """
        <div class="section">
            <h2>Disclaimer</h2>
            <p>This report is generated for educational purposes only. Cryptocurrency investments are subject to high market risk. 
            Past performance is not indicative of future results. The predictions are based on historical data and may not accurately 
            reflect future market conditions.</p>
        </div>
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    import weasyprint
    weasyprint.HTML(string=html).write_pdf(buffer)
    
    buffer.seek(0)
    return buffer

# Main content area
with st.spinner(f"Loading data for {selected_symbol}..."):
    # Get historical data
    historical_data = load_historical_data(selected_symbol, selected_interval, selected_time_range)
    
    # Get current price
    current_price = get_current_price(selected_symbol)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Prediction", "📈 Technical Analysis", "📝 Report"])

# Tab 1: Overview
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # Display current price
    if current_price:
        st.markdown(f"""
        <div class="info-box">
        <h2>Current Price for {selected_symbol}</h2>
        <h1>${current_price:.2f}</h1>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Could not fetch current price.")
    
    # Display historical data
    st.markdown('<h2 class="sub-header">Historical Price Data</h2>', unsafe_allow_html=True)
    
    if historical_data is not None and not historical_data.empty:
        # Create interactive plot with Plotly
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                name="Close Price",
                line=dict(color='royalblue', width=2)
            )
        )
        
        # Add volume bars on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data['Volume'],
                name="Volume",
                marker=dict(color='rgba(58, 71, 80, 0.3)')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_symbol} Historical Data ({selected_interval_name} intervals, {selected_time_range_name} range)",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        st.markdown('<h2 class="sub-header">Price Statistics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Highest Price", f"${historical_data['High'].max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Lowest Price", f"${historical_data['Low'].min():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            price_change = historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[0]
            price_change_pct = (price_change / historical_data['Close'].iloc[0]) * 100
            st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg. Daily Volume", f"{historical_data['Volume'].mean():.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display recent data in a table
        st.markdown('<h2 class="sub-header">Recent Data</h2>', unsafe_allow_html=True)
        st.dataframe(historical_data.tail(10).style.format({
            'Open': '${:.2f}',
            'High': '${:.2f}',
            'Low': '${:.2f}',
            'Close': '${:.2f}',
            'Adj Close': '${:.2f}',
            'Volume': '{:,.0f}'
        }))
    else:
        st.warning(f"Could not fetch historical data for {selected_symbol}.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Prediction
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Price Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        predict_button = st.button("Generate Prediction", use_container_width=True)
    
    with col1:
        st.markdown(f"Predicting next {prediction_days} day(s) for {selected_symbol}")
    
    if predict_button:
        with st.spinner("Generating prediction..."):
            predicted_price, error_message = predict_next_price(selected_symbol, prediction_days)
        
        if predicted_price is not None:
            # Handle both single value and list of predictions
            if isinstance(predicted_price, list):
                # Multi-day prediction
                st.markdown(f"""
                <div class="prediction-box">
                <h2>Predicted Prices for Next {len(predicted_price)} Days</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a dataframe for the predictions
                dates = [datetime.now() + timedelta(days=i) for i in range(1, len(predicted_price) + 1)]
                pred_df = pd.DataFrame({
                    'Date': dates,
                    'Predicted Price': predicted_price
                })
                
                # Display predictions in a table
                st.dataframe(pred_df.style.format({
                    'Predicted Price': '${:.2f}'
                }))
                
                # Create a line chart for the predictions
                fig = px.line(
                    pred_df, 
                    x='Date', 
                    y='Predicted Price',
                    title=f"Predicted Prices for {selected_symbol} - Next {len(predicted_price)} Days",
                    labels={'Predicted Price': 'Price (USD)', 'Date': 'Date'}
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate change from current price for the first prediction
                if current_price:
                    first_price_diff = predicted_price[0] - current_price
                    first_price_diff_pct = (first_price_diff / current_price) * 100
                    direction = "increase" if first_price_diff > 0 else "decrease"
                    
                    # Alert if the predicted change exceeds the threshold
                    if abs(first_price_diff_pct) >= alert_threshold:
                        st.markdown(f"""
                        <div class="alert-box">
                        <h3>⚠️ Significant Price Movement Alert</h3>
                        <p>The model predicts a significant {direction} of {abs(first_price_diff_pct):.2f}% for tomorrow.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Single day prediction
                # Calculate change from current price
                if current_price:
                    price_diff = predicted_price - current_price
                    price_diff_pct = (price_diff / current_price) * 100
                    direction = "increase" if price_diff > 0 else "decrease"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                    <h2>Predicted Next Price</h2>
                    <h1>${predicted_price:.2f}</h1>
                    <p>Expected {direction} of ${abs(price_diff):.2f} ({abs(price_diff_pct):.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Alert if the predicted change exceeds the threshold
                    if abs(price_diff_pct) >= alert_threshold:
                        st.markdown(f"""
                        <div class="alert-box">
                        <h3>⚠️ Significant Price Movement Alert</h3>
                        <p>The model predicts a significant {direction} of {abs(price_diff_pct):.2f}%.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add a simple visualization of the prediction
                    if historical_data is not None and not historical_data.empty:
                        # Create a new dataframe for visualization
                        last_date = historical_data.index[-1]
                        next_date = last_date + timedelta(days=1)  # Assuming daily prediction
                        
                        # Create a dataframe with historical and predicted data
                        pred_df = historical_data[['Close']].copy()
                        pred_df.loc[next_date] = predicted_price
                        
                        # Plot the data
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(
                            go.Scatter(
                                x=pred_df.index[:-1],
                                y=pred_df['Close'][:-1],
                                name="Historical Price",
                                line=dict(color='royalblue', width=2)
                            )
                        )
                        
                        # Prediction
                        fig.add_trace(
                            go.Scatter(
                                x=[pred_df.index[-2], pred_df.index[-1]],
                                y=[pred_df['Close'][-2], pred_df['Close'][-1]],
                                name="Prediction",
                                line=dict(color='green', width=3, dash='dash')
                            )
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title="Price Prediction Visualization",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box">
                    <h2>Predicted Next Price</h2>
                    <h1>${predicted_price:.2f}</h1>
                    <p>Current price unavailable for comparison</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
            <h3>Prediction Error</h3>
            <p>{error_message}</p>
            <p>This may be due to insufficient historical data or model issues.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Volatility Analysis
    st.markdown('<h2 class="sub-header">Volatility Analysis</h2>', unsafe_allow_html=True)
    
    if historical_data is not None and not historical_data.empty:
        # Calculate daily returns
        if 'Close' in historical_data.columns:
            historical_data['Daily Return'] = historical_data['Close'].pct_change() * 100
            
            # Calculate rolling volatility (20-day standard deviation of returns)
            historical_data['Volatility'] = historical_data['Daily Return'].rolling(window=20).std()
            
            # Create volatility plot
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Volatility'],
                    name="20-Day Volatility",
                    line=dict(color='purple', width=2)
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Price Volatility Over Time (20-Day Rolling Standard Deviation)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recent volatility
            recent_volatility = historical_data['Volatility'].iloc[-1]
            avg_volatility = historical_data['Volatility'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Current Volatility", f"{recent_volatility:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Volatility", f"{avg_volatility:.2f}%", 
                         f"{recent_volatility - avg_volatility:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Volatility interpretation
            if recent_volatility < avg_volatility * 0.5:
                volatility_message = "Very Low: The market is currently very stable compared to historical patterns."
            elif recent_volatility < avg_volatility * 0.8:
                volatility_message = "Low: The market is showing below-average volatility."
            elif recent_volatility < avg_volatility * 1.2:
                volatility_message = "Normal: The market is showing typical volatility levels."
            elif recent_volatility < avg_volatility * 2:
                volatility_message = "High: The market is more volatile than usual."
            else:
                volatility_message = "Very High: The market is experiencing extreme volatility."
            
            st.info(f"**Volatility Assessment**: {volatility_message}")
    else:
        st.warning("Insufficient data for volatility analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Technical Analysis
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Technical Analysis</h2>', unsafe_allow_html=True)
    
    if historical_data is not None and not historical_data.empty and show_technical_indicators:
        # Calculate technical indicators
        tech_data = calculate_technical_indicators(historical_data)
        
        # Create tabs for different technical indicators
        tech_tab1, tech_tab2, tech_tab3 = st.tabs(["Moving Averages", "RSI & MACD", "Bollinger Bands"])
        
        # Tab 1: Moving Averages
        with tech_tab1:
            st.markdown('<h3>Moving Averages</h3>', unsafe_allow_html=True)
            
            # Create Moving Averages plot
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['Close'],
                    name="Close Price",
                    line=dict(color='black', width=2)
                )
            )
            
            # Add Moving Averages
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MA20'],
                    name="20-Day MA",
                    line=dict(color='blue', width=1.5)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MA50'],
                    name="50-Day MA",
                    line=dict(color='orange', width=1.5)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MA200'],
                    name="200-Day MA",
                    line=dict(color='red', width=1.5)
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Moving Averages Analysis",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Moving Average Analysis
            st.markdown('<h4>Moving Average Analysis</h4>', unsafe_allow_html=True)
            
            # Get the latest values
            latest_close = tech_data['Close'].iloc[-1]
            latest_ma20 = tech_data['MA20'].iloc[-1]
            latest_ma50 = tech_data['MA50'].iloc[-1]
            latest_ma200 = tech_data['MA200'].iloc[-1]
            
            # Determine market trend based on MA relationships
            if latest_ma20 > latest_ma50 > latest_ma200:
                trend = "Strong Uptrend"
                trend_description = "All moving averages are aligned in an upward direction, suggesting a strong bullish trend."
            elif latest_ma20 > latest_ma50 and latest_ma50 < latest_ma200:
                trend = "Potential Reversal (Bullish)"
                trend_description = "Short-term MA is above medium-term MA but below long-term MA, suggesting a potential bullish reversal."
            elif latest_ma20 < latest_ma50 and latest_ma50 > latest_ma200:
                trend = "Weakening Uptrend"
                trend_description = "Short-term MA is below medium-term MA but above long-term MA, suggesting a weakening uptrend."
            elif latest_ma20 < latest_ma50 < latest_ma200:
                trend = "Strong Downtrend"
                trend_description = "All moving averages are aligned in a downward direction, suggesting a strong bearish trend."
            else:
                trend = "Mixed Signals"
                trend_description = "Moving averages are showing mixed signals, suggesting a sideways or uncertain market."
            
            # Price position relative to MAs
            if latest_close > latest_ma20:
                price_position = "Above 20-Day MA"
                if latest_close > latest_ma50:
                    price_position += " and 50-Day MA"
                    if latest_close > latest_ma200:
                        price_position += " and 200-Day MA, suggesting strong bullish momentum."
                    else:
                        price_position += ", but below 200-Day MA, suggesting medium-term bullish momentum."
                else:
                    price_position += ", but below 50-Day MA, suggesting short-term bullish momentum only."
            else:
                price_position = "Below 20-Day MA"
                if latest_close < latest_ma50:
                    price_position += " and 50-Day MA"
                    if latest_close < latest_ma200:
                        price_position += " and 200-Day MA, suggesting strong bearish momentum."
                    else:
                        price_position += ", but above 200-Day MA, suggesting medium-term bearish momentum."
                else:
                    price_position += ", but above 50-Day MA, suggesting short-term bearish momentum only."
            
            st.markdown(f"""
            <div class="info-box">
            <p><strong>Current Trend:</strong> {trend}</p>
            <p>{trend_description}</p>
            <p><strong>Price Position:</strong> {price_position}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tab 2: RSI & MACD
        with tech_tab2:
            st.markdown('<h3>RSI & MACD</h3>', unsafe_allow_html=True)
            
            # Create subplot with 2 rows
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, 
                               subplot_titles=("Price", "RSI & MACD"))
            
            # Add price to first row
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['Close'],
                    name="Close Price",
                    line=dict(color='black', width=2)
                ),
                row=1, col=1
            )
            
            # Add RSI to second row
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['RSI'],
                    name="RSI",
                    line=dict(color='purple', width=1.5)
                ),
                row=2, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=[tech_data.index[0], tech_data.index[-1]],
                    y=[70, 70],
                    name="Overbought (70)",
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[tech_data.index[0], tech_data.index[-1]],
                    y=[30, 30],
                    name="Oversold (30)",
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=2, col=1
            )
            
            # Add MACD to second row
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MACD'],
                    name="MACD",
                    line=dict(color='blue', width=1.5)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['Signal'],
                    name="Signal Line",
                    line=dict(color='orange', width=1.5)
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis1_title="Price (USD)",
                yaxis2_title="Value"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI & MACD Analysis
            st.markdown('<h4>RSI & MACD Analysis</h4>', unsafe_allow_html=True)
            
            # Get the latest values
            latest_rsi = tech_data['RSI'].iloc[-1]
            latest_macd = tech_data['MACD'].iloc[-1]
            latest_signal = tech_data['Signal'].iloc[-1]
            
            # RSI Analysis
            if latest_rsi > 70:
                rsi_status = "Overbought"
                rsi_description = "The RSI is above 70, suggesting the asset may be overbought and could be due for a price correction."
            elif latest_rsi < 30:
                rsi_status = "Oversold"
                rsi_description = "The RSI is below 30, suggesting the asset may be oversold and could be due for a price rebound."
            else:
                rsi_status = "Neutral"
                rsi_description = f"The RSI is at {latest_rsi:.2f}, which is within the neutral range (30-70)."
            
            # MACD Analysis
            if latest_macd > latest_signal:
                if latest_macd > 0:
                    macd_status = "Strong Bullish"
                    macd_description = "MACD is above the signal line and positive, suggesting strong bullish momentum."
                else:
                    macd_status = "Bullish"
                    macd_description = "MACD is above the signal line but negative, suggesting potential bullish momentum building."
            else:
                if latest_macd < 0:
                    macd_status = "Strong Bearish"
                    macd_description = "MACD is below the signal line and negative, suggesting strong bearish momentum."
                else:
                    macd_status = "Bearish"
                    macd_description = "MACD is below the signal line but positive, suggesting potential bearish momentum building."
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-box">
                <h4>RSI Analysis</h4>
                <p><strong>Current RSI:</strong> {latest_rsi:.2f}</p>
                <p><strong>Status:</strong> {rsi_status}</p>
                <p>{rsi_description}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                <h4>MACD Analysis</h4>
                <p><strong>MACD:</strong> {latest_macd:.4f}</p>
                <p><strong>Signal:</strong> {latest_signal:.4f}</p>
                <p><strong>Status:</strong> {macd_status}</p>
                <p>{macd_description}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 3: Bollinger Bands
        with tech_tab3:
            st.markdown('<h3>Bollinger Bands</h3>', unsafe_allow_html=True)
            
            # Create Bollinger Bands plot
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['Close'],
                    name="Close Price",
                    line=dict(color='black', width=2)
                )
            )
            
            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['upper_band'],
                    name="Upper Band",
                    line=dict(color='red', width=1.5)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['MA20'],
                    name="20-Day MA (Middle Band)",
                    line=dict(color='blue', width=1.5)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['lower_band'],
                    name="Lower Band",
                    line=dict(color='green', width=1.5)
                )
            )
            
            # Fill the area between the bands
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['upper_band'],
                    fill=None,
                    line=dict(color='rgba(0,0,0,0)')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data['lower_band'],
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.2)',
                    line=dict(color='rgba(0,0,0,0)')
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Bollinger Bands Analysis",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bollinger Bands Analysis
            st.markdown('<h4>Bollinger Bands Analysis</h4>', unsafe_allow_html=True)
            
            # Get the latest values
            latest_close = tech_data['Close'].iloc[-1]
            latest_upper = tech_data['upper_band'].iloc[-1]
            latest_middle = tech_data['MA20'].iloc[-1]
            latest_lower = tech_data['lower_band'].iloc[-1]
            
            # Calculate band width and %B
            band_width = (latest_upper - latest_lower) / latest_middle
            percent_b = (latest_close - latest_lower) / (latest_upper - latest_lower) if (latest_upper - latest_lower) != 0 else 0.5
            
            # Bollinger Band Analysis
            if latest_close > latest_upper:
                bb_status = "Above Upper Band"
                bb_description = "Price is above the upper Bollinger Band, suggesting the asset may be overbought. This could indicate a potential reversal or continuation of a strong uptrend."
            elif latest_close < latest_lower:
                bb_status = "Below Lower Band"
                bb_description = "Price is below the lower Bollinger Band, suggesting the asset may be oversold. This could indicate a potential reversal or continuation of a strong downtrend."
            else:
                bb_status = "Within Bands"
                if percent_b > 0.8:
                    bb_description = "Price is near the upper band, suggesting potential resistance or overbought conditions approaching."
                elif percent_b < 0.2:
                    bb_description = "Price is near the lower band, suggesting potential support or oversold conditions approaching."
                else:
                    bb_description = "Price is within the middle range of the Bollinger Bands, suggesting relatively neutral conditions."
            
            # Band Width Analysis
            if band_width > 0.1:
                volatility_status = "High"
                volatility_description = "The Bollinger Bands are wide, indicating high market volatility. This could suggest a potential continuation of the current trend or an upcoming significant price movement."
            elif band_width < 0.05:
                volatility_status = "Low"
                volatility_description = "The Bollinger Bands are narrow, indicating low market volatility. This often precedes a significant price movement as volatility tends to cycle between high and low periods."
            else:
                volatility_status = "Moderate"
                volatility_description = "The Bollinger Bands show moderate width, indicating average market volatility."
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="info-box">
                <h4>Price Position</h4>
                <p><strong>Status:</strong> {bb_status}</p>
                <p><strong>%B:</strong> {percent_b:.2f}</p>
                <p>{bb_description}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                <h4>Volatility Analysis</h4>
                <p><strong>Band Width:</strong> {band_width:.4f}</p>
                <p><strong>Volatility:</strong> {volatility_status}</p>
                <p>{volatility_description}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        if not show_technical_indicators:
            st.info("Technical indicators are disabled. Enable them in the Advanced Settings panel.")
        else:
            st.warning("Insufficient data for technical analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Report
with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Analysis Report</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Generate a comprehensive analysis report for the selected cryptocurrency. 
    The report includes current price, historical data summary, predictions, and volatility analysis.
    """)
    
    # Add prediction to the report if not already generated
    if 'predicted_price' not in locals():
        with st.spinner("Generating prediction for report..."):
            predicted_price, _ = predict_next_price(selected_symbol, prediction_days)
    
    # Generate and provide download link for the report
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            try:
                report_buffer = generate_report(
                    selected_symbol, 
                    historical_data, 
                    current_price, 
                    predicted_price,
                    None  # Volatility data is included in historical_data
                )
                
                # Create download link
                b64 = base64.b64encode(report_buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="{selected_symbol}_analysis_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("Report generated successfully! Click the link above to download.")
            except Exception as e:
                st.error(f"Error generating report: {e}")
    
    # Report preview
    st.markdown('<h3>Report Preview</h3>', unsafe_allow_html=True)
    
    if historical_data is not None and not historical_data.empty:
        # Create a summary table
        summary_data = {
            "Metric": ["Current Price", "Highest Price (Period)", "Lowest Price (Period)", "Average Price", "Price Change"],
            "Value": [
                f"${current_price:.2f}" if current_price else "N/A",
                f"${historical_data['High'].max():.2f}",
                f"${historical_data['Low'].min():.2f}",
                f"${historical_data['Close'].mean():.2f}",
                f"${historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[0]:.2f} ({(historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[0]) / historical_data['Close'].iloc[0] * 100:.2f}%)"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Add prediction preview if available
        if 'predicted_price' in locals() and predicted_price is not None:
            if isinstance(predicted_price, list):
                st.markdown(f"**Prediction:** Next {len(predicted_price)} days forecast included in report")
            else:
                st.markdown(f"**Prediction:** Next day forecast of ${predicted_price:.2f} included in report")
        
        # Add volatility preview if available
        if 'Volatility' in historical_data.columns:
            recent_volatility = historical_data['Volatility'].iloc[-1]
            avg_volatility = historical_data['Volatility'].mean()
            st.markdown(f"**Volatility Analysis:** Current volatility at {recent_volatility:.2f}% vs. average of {avg_volatility:.2f}%")
    else:
        st.warning("Insufficient data for report preview.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
---
<p style="text-align: center;">Cryptocurrency Price Predictor | Powered by LSTM Neural Networks</p>
<p style="text-align: center;">Disclaimer: This tool is for educational purposes only. Cryptocurrency investments are subject to high market risk.</p>
""", unsafe_allow_html=True)
