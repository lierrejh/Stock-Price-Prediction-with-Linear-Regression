import yfinance as yf
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy, matplotlib
from sklearn.preprocessing import StandardScaler


# Download historical stock data for compan(y/ies)
def download_stock_data(ticker):
    stock_data = yf.download(ticker, period='1y')
    df = stock_data[['Close', 'Volume']].copy()
    return df

def predictor_variables(df):
    df['Returns_2d'] = df['Close'].pct_change(2)
    df['MA_5d'] = df['Close'].rolling(window=5).mean()
    # df['Volume_1d'] = df['Volume'].pct_change(1)
    df['Volume_vs_Avg'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    return df.dropna()

def prediction_target(df):
    df['Target'] = df['Close'].shift(-1)
    return df.dropna()

def train_test(df, test_size=0.2):
    X = df[['Returns_2d', 'MA_5d', 'Volume_vs_Avg']]
    y = df['Target']    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # maintain time order
)
    return X_train, X_test, y_train, y_test

def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    return model, train_predictions, test_predictions,  y_train, y_test

def evaluate_model(y_test, test_predictions):
    mse = sklearn.metrics.mean_squared_error(y_test, test_predictions)
    mae = sklearn.metrics.mean_absolute_error(y_test, test_predictions)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    return mse, mae

def plot_predictions(y_test, test_predictions):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
    plt.plot(y_test.index, test_predictions, label='Predicted Prices', color='red')
    plt.title('Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def evaluate_baseline(y_test, X_test):
    naive_preds = stock_data.loc[y_test.index, 'Close']
    naive_mae = sklearn.metrics.mean_absolute_error(y_test, naive_preds)
    print(f'Naive MAE (Predicting Moving Average): {naive_mae}')
    return naive_mae

def scale_features(df):
    features = ['Returns_2d', 'MA_5d', 'Volume_vs_Avg']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

if __name__ == "__main__":
    ticker = 'AAPL'  # Example ticker
    stock_data = download_stock_data(ticker)
    stock_data = predictor_variables(stock_data)
    stock_data = prediction_target(stock_data)

    stock_data, scaler = scale_features(stock_data)

    X_train, X_test, y_train, y_test = train_test(stock_data)
    model, train_predictions, test_predictions, y_train, y_test = linear_regression(X_train, y_train, X_test, y_test)
    mse, mae = evaluate_model(y_test, test_predictions)
    plot_predictions(y_test, test_predictions)
    naive_mae = evaluate_baseline(y_test, X_test)
    print(f'Linear Regression Model Coefficients: {model.coef_}')