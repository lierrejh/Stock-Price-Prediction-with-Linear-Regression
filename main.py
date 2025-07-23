import yfinance as yf
import pandas as pd
import sklearn
import numpy, matplotlib

# Download historical stock data for compan(y/ies)
def download_stock_data(ticker):
    stock_data = yf.download(ticker, period='1y')
    df = stock_data[['Close', 'Volume']].copy()
    return df

def predictor_variables(df):
    df['Returns_1d'] = df['Close'].pct_change(1)
    df['Returns_2d'] = df['Close'].pct_change(2)
    df['MA_5d'] = df['Close'].rolling(window=5).mean()
    df['Volume_1d'] = df['Volume'].pct_change(1)
    df['Volume_vs_Avg'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    return df.dropna()

def prediction_target(df):
    df['Target'] = df['Close'].shift(-1)
    return df.dropna()

def train_test_split(df, test_size=0.2):
    train_size = int(len(df) * (1 - test_size))
    train_set = df[:train_size]
    test_set = df[train_size:]
    return train_set, test_set



if __name__ == "__main__":
    ticker = 'AAPL'
    stock_data = download_stock_data(ticker)
    stock_data = predictor_variables(stock_data)
    stock_data = prediction_target(stock_data)
