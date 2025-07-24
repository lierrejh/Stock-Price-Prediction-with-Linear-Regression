import yfinance as yf
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Data Loading

def download_stock_data(ticker):
    """Download 1 year of daily stock data from Yahoo Finance."""
    stock_data = yf.download(ticker, period='1y')
    df = stock_data[['Close', 'Volume']].copy()
    return df

# Feature Engineering

def predictor_variables(df):
    """Create predictive features."""
    df['Returns_1d'] = df['Close'].pct_change(1)
    df['Returns_2d'] = df['Close'].pct_change(2)
    df['Returns_5d'] = df['Close'].pct_change(5)
    df['MA_10d'] = df['Close'].rolling(window=10).mean()
    df['MA_5d'] = df['Close'].rolling(window=5).mean()
    df['Close_to_MA5'] = df['Close'] / df['Close'].rolling(window=5).mean()
    df['LogVolume'] = numpy.log(df['Volume'] + 1)
    df['Returns_1d_x_Close_to_MA5'] = df['Returns_1d'] * df['Close_to_MA5']
    df['MA_ratio'] = df['MA_5d'] / df['MA_10d']
    df['Returns_2d_x_MA10'] = df['Returns_2d'] * df['MA_10d']
    return df.dropna()

def prediction_target(df):
    """Define target as next-day return."""
    df['Target'] = df['Close'].pct_change().shift(-1)
    return df.dropna()

# Preprocessing

def scale_features(df):
    """Standardise feature columns."""
    features = [
        'Returns_1d', 'Returns_2d', 'Returns_5d',
        'MA_5d', 'MA_10d', 'Close_to_MA5',
        'LogVolume', 'Returns_1d_x_Close_to_MA5', 'MA_ratio', 'Returns_2d_x_MA10'
    ]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

# Training

def linear_regression(X_train, y_train, X_test, y_test):
    """Train Linear Regression model and return predictions."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    return model, train_predictions, test_predictions, y_train, y_test

# Evaluation

def evaluate_model(y_test, test_predictions):
    """Print and return MAE and MSE."""
    mse = sklearn.metrics.mean_squared_error(y_test, test_predictions)
    mae = sklearn.metrics.mean_absolute_error(y_test, test_predictions)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    return mse, mae

def evaluate_baseline(y_test, stock_data):
    """Compare to naive baseline: predict today's return as tomorrow's."""
    naive_preds = stock_data.loc[y_test.index, 'Returns_1d']
    naive_mae = sklearn.metrics.mean_absolute_error(y_test, naive_preds)
    print(f'Naive MAE (Predicting Today\'s Return): {naive_mae}')
    return naive_mae

# Rolling Window Backtest

def rolling_window_evaluation(df, feature_cols, target_col, train_size=150, test_size=1):
    """Perform rolling window backtest and return predictions and actuals."""
    errors = []
    predictions = []
    actuals = []
    indices = []

    for start in range(0, len(df) - train_size - test_size):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]

        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        predictions.extend(pred)
        actuals.extend(y_test.values)
        indices.extend(y_test.index)

        error = sklearn.metrics.mean_absolute_error(y_test, pred)
        errors.append(error)

    avg_mae = sum(errors) / len(errors)
    print(f"\nRolling Window MAE (avg across {len(errors)} tests): {avg_mae:.6f}")
    
    return indices, actuals, predictions

# Main Execution

if __name__ == "__main__":
    ticker = 'AAPL'  

    # Download and process data
    stock_data = download_stock_data(ticker)
    stock_data = predictor_variables(stock_data)
    stock_data = prediction_target(stock_data)
    stock_data, scaler = scale_features(stock_data)

    # Correlation-based feature filtering
    corr_matrix = stock_data.corr()
    target_corr = corr_matrix['Target'].drop('Target')
    selected_features = target_corr[target_corr.abs() > 0.05].index.tolist()

    print("\nSelected features based on correlation with Target:")
    for feature in selected_features:
        print(f"{feature}: {target_corr[feature]:.4f}")

    # Reconstruct X/y with filtered features
    X = stock_data[selected_features]
    y = stock_data['Target']

    # Standardise selected features (again, post-filter)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine back into a DataFrame
    scaled_df = pd.DataFrame(X_scaled, columns=selected_features, index=stock_data.index)
    scaled_df['Target'] = y

    # Rolling window evaluation
    indices, actuals, predictions = rolling_window_evaluation(
        scaled_df,
        feature_cols=selected_features,
        target_col='Target',
        train_size=60,
        test_size=1
    )

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.plot(indices, actuals, label="Actual Returns", color="blue")
    plt.plot(indices, predictions, label="Predicted Returns", color="red")
    plt.title("Rolling Window Prediction Performance")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()
