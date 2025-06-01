#!/usr/bin/env python3

import pandas as pd
from datetime import datetime
import os
import sys

# Add the path for the ApiClient
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

# Constants
DATA_DIR = "data"
DEFAULT_SYMBOL = "BTC-USD"  # Yahoo Finance format for Bitcoin
DEFAULT_INTERVAL = "1d"     # Daily interval
DEFAULT_RANGE = "5y"        # 5 years of data


def get_historical_data_yahoo(symbol, interval, range_str):
    """Fetches historical data from Yahoo Finance API and returns a pandas DataFrame."""
    client = ApiClient()
    print(f"Fetching {interval} data for {symbol} over the last {range_str} from Yahoo Finance...")

    try:
        # Call the Yahoo Finance API
        response = client.call_api(
            'YahooFinance/get_stock_chart',
            query={
                'symbol': symbol,
                'interval': interval,
                'range': range_str,
                'includeAdjustedClose': 'true' # Include adjusted close if needed, though less relevant for crypto
            }
        )

        # Check for errors in the response
        if not response or response.get('chart', {}).get('error'):
            error_message = response.get('chart', {}).get('error') or 'Unknown API error'
            print(f"Error fetching data for {symbol} from Yahoo Finance: {error_message}")
            return None

        # Extract data
        chart_result = response.get('chart', {}).get('result', [])
        if not chart_result:
            print(f"No data found for {symbol} in the response.")
            return None

        data = chart_result[0]
        timestamps = data.get('timestamp', [])
        indicators = data.get('indicators', {})
        quote = indicators.get('quote', [{}])[0]
        adjclose = indicators.get('adjclose', [{}])[0].get('adjclose', []) if indicators.get('adjclose') else []

        if not timestamps or not quote.get('open'):
            print(f"Incomplete data received for {symbol}.")
            return None

        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': pd.to_datetime(timestamps, unit='s'),
            'Open': quote.get('open'),
            'High': quote.get('high'),
            'Low': quote.get('low'),
            'Close': quote.get('close'),
            'Volume': quote.get('volume')
        })

        # Add Adjusted Close if available
        if adjclose and len(adjclose) == len(df):
             df['Adj Close'] = adjclose
        else:
             df['Adj Close'] = df['Close'] # Fallback if adjclose is missing/mismatched

        # Set Timestamp as index
        df.set_index('Timestamp', inplace=True)

        # Convert columns to numeric, coercing errors
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaN values introduced by coercion or missing data
        df.dropna(inplace=True)

        print(f"Data processed into DataFrame with shape: {df.shape}")
        return df

    except Exception as e:
        print(f"An unexpected error occurred while fetching/processing data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_data_to_csv(df, symbol, interval):
    """Saves the DataFrame to a CSV file in the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Sanitize symbol for filename if needed (though BTC-USD is fine)
    safe_symbol = symbol.replace('/', '_') # Example sanitization
    filename = f"{safe_symbol}_{interval}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
    return filepath

if __name__ == "__main__":
    symbol_to_fetch = DEFAULT_SYMBOL
    interval_to_fetch = DEFAULT_INTERVAL
    range_to_fetch = DEFAULT_RANGE

    historical_df = get_historical_data_yahoo(symbol_to_fetch, interval_to_fetch, range_to_fetch)

    if historical_df is not None and not historical_df.empty:
        saved_path = save_data_to_csv(historical_df, symbol_to_fetch, interval_to_fetch)
        print(f"Successfully fetched and saved data to {saved_path}")
    else:
        print(f"Failed to fetch or process data for {symbol_to_fetch}.")




def get_real_time_price_yahoo(symbol):
    """Fetches the latest available price from Yahoo Finance API."""
    client = ApiClient()
    print(f"\nFetching real-time price for {symbol} from Yahoo Finance...")

    try:
        # Use a short range to get recent data, relying on meta.regularMarketPrice
        response = client.call_api(
            'YahooFinance/get_stock_chart',
            query={
                'symbol': symbol,
                'range': '1d', # Get data for the current day
                'interval': '5m' # Use a small interval; meta price should be current regardless
            }
        )

        if not response or response.get('chart', {}).get('error'):
            error_message = response.get('chart', {}).get('error') or 'Unknown API error'
            print(f"Error fetching real-time data for {symbol} from Yahoo Finance: {error_message}")
            # Attempt to extract meta price even if there's a chart error
            meta = response.get('chart', {}).get('result', [{}])[0].get('meta', {}) if response else {}
            if meta and 'regularMarketPrice' in meta:
                 price = meta['regularMarketPrice']
                 timestamp = meta.get('regularMarketTime')
                 dt_object = datetime.fromtimestamp(timestamp) if timestamp else "N/A"
                 print(f"Using potentially stale regularMarketPrice from meta due to error: {price} at {dt_object}")
                 return price
            else:
                 print("Could not retrieve real-time price from meta after error.")
                 return None

        chart_result = response.get('chart', {}).get('result', [])
        if not chart_result:
            print(f"No chart result found for {symbol}.")
            return None

        meta = chart_result[0].get('meta', {})
        if meta and 'regularMarketPrice' in meta:
            price = meta['regularMarketPrice']
            timestamp = meta.get('regularMarketTime')
            dt_object = datetime.fromtimestamp(timestamp) if timestamp else "N/A"
            print(f"Latest price (regularMarketPrice) for {symbol}: {price} at {dt_object}")
            return price
        else:
            # Fallback: Get the last closing price from the indicators if meta price is unavailable
            print("regularMarketPrice not found in meta, attempting fallback to last close price.")
            indicators = chart_result[0].get('indicators', {})
            quote = indicators.get('quote', [{}])[0]
            close_prices = quote.get('close', [])
            timestamps = chart_result[0].get('timestamp', [])

            if close_prices and timestamps:
                # Filter out potential null values at the end
                valid_indices = [i for i, p in enumerate(close_prices) if p is not None]
                if not valid_indices:
                    print(f"No valid close prices found for {symbol}.")
                    return None
                last_valid_index = valid_indices[-1]
                last_price = close_prices[last_valid_index]
                last_timestamp = timestamps[last_valid_index]
                dt_object = datetime.fromtimestamp(last_timestamp) if last_timestamp else "N/A"
                print(f"Using last close price for {symbol}: {last_price} at {dt_object}")
                # Consider adding a check here for how recent the timestamp is
                return last_price
            else:
                print(f"Could not extract real-time price for {symbol} from response indicators.")
                return None

    except Exception as e:
        print(f"An unexpected error occurred while fetching real-time price for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Modify the main block to test the new function
if __name__ == "__main__":
    symbol_to_fetch = DEFAULT_SYMBOL
    interval_to_fetch = DEFAULT_INTERVAL
    range_to_fetch = DEFAULT_RANGE

    print("--- Fetching Historical Data ---")
    historical_df = get_historical_data_yahoo(symbol_to_fetch, interval_to_fetch, range_to_fetch)

    if historical_df is not None and not historical_df.empty:
        saved_path = save_data_to_csv(historical_df, symbol_to_fetch, interval_to_fetch)
        print(f"Successfully fetched and saved historical data to {saved_path}")
    else:
        print(f"Failed to fetch or process historical data for {symbol_to_fetch}.")

    print("\n--- Fetching Real-Time Price ---")
    real_time_price = get_real_time_price_yahoo(symbol_to_fetch)
    if real_time_price is not None:
        print(f"Current price for {symbol_to_fetch}: {real_time_price}")
    else:
        print(f"Failed to fetch real-time price for {symbol_to_fetch}.")
