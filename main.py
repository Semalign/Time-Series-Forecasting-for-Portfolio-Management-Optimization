"""
Main pipeline for GMF Investments Time Series Forecasting & Portfolio Optimization
"""

import os
from dotenv import load_dotenv

# Import modules from src/
from src.data_fetch import fetch_data
from src.data_preprocessing import preprocess_data
from src.eda import run_eda
from src.models_arima import run_arima_forecast
from src.models_lstm import run_lstm_forecast
from src.forecast_analysis import analyze_forecast
from src.portfolio_optimization import optimize_portfolio
from src.backtesting import backtest_strategy

# Load environment variables
load_dotenv()

def main():
    print("\n📊 GMF Investments Pipeline Started...\n")

    # Step 1: Fetch Data
    print("Step 1: Fetching data from YFinance...")
    tickers = ['TSLA', 'BND', 'SPY']
    raw_data = fetch_data(tickers, start="2015-07-01", end="2025-07-31")

    # Step 2: Preprocess Data
    print("\nStep 2: Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    # Step 3: Exploratory Data Analysis
    print("\nStep 3: Running EDA...")
    run_eda(processed_data)

    # Step 4: Forecasting with ARIMA
    print("\nStep 4: Forecasting with ARIMA...")
    arima_forecast, arima_metrics = run_arima_forecast(processed_data['TSLA'])

    # Step 5: Forecasting with LSTM
    print("\nStep 5: Forecasting with LSTM...")
    lstm_forecast, lstm_metrics = run_lstm_forecast(processed_data['TSLA'])

    # Step 6: Compare models & pick best
    print("\nStep 6: Comparing models...")
    if lstm_metrics['RMSE'] < arima_metrics['RMSE']:
        best_model_forecast = lstm_forecast
        print("✅ Selected LSTM as best model")
    else:
        best_model_forecast = arima_forecast
        print("✅ Selected ARIMA as best model")

    # Step 7: Analyze forecast
    print("\nStep 7: Analyzing forecast...")
    analyze_forecast(best_model_forecast, processed_data['TSLA'])

    # Step 8: Portfolio Optimization
    print("\nStep 8: Optimizing portfolio...")
    optimal_portfolio = optimize_portfolio(best_model_forecast, processed_data)

    # Step 9: Backtesting
    print("\nStep 9: Backtesting strategy...")
    backtest_strategy(optimal_portfolio, processed_data)

    print("\n🎯 Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
