import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import combinations


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
    df['LogVolume'] = np.log(df['Volume'] + 1)
    df['Returns_1d_x_Close_to_MA5'] = df['Returns_1d'] * df['Close_to_MA5']
    df['MA_ratio'] = df['MA_5d'] / df['MA_10d']
    df['Returns_2d_x_MA10'] = df['Returns_2d'] * df['MA_10d']
    df['RollingVol_5d'] = df['Volume'].rolling(window=5).std()
    df['Volume_vs_Avg'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    return df.dropna()

def prediction_target(df):
    """Define target as next-day return."""
    df['Target'] = df['Close'].pct_change().shift(-1)
    return df.dropna()

# Preprocessing

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


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
    
    return indices, actuals, predictions

# Main Execution

if __name__ == "__main__":
    ticker = 'AAPL'
    stock_data = download_stock_data(ticker)
    stock_data = predictor_variables(stock_data)
    stock_data = prediction_target(stock_data)

    candidate_features = [
        'Returns_1d', 'Returns_2d', 'Returns_5d',
        'MA_5d', 'MA_10d', 'Close_to_MA5',
        'LogVolume', 'Returns_1d_x_Close_to_MA5',
        'MA_ratio', 'Returns_2d_x_MA10',
        'RollingVol_5d', 'Volume_vs_Avg'
    ]

    stock_data = scale_features(stock_data, candidate_features)

# Uncomment the following lines to run the exhaustive feature combination search
    # # Full list of candidate features (add more here as needed)
    # candidate_features = [
    #     'Returns_1d', 'Returns_2d', 'Returns_5d',
    #     'MA_5d', 'MA_10d', 'Close_to_MA5',
    #     'LogVolume', 'Returns_1d_x_Close_to_MA5',
    #     'MA_ratio', 'Returns_2d_x_MA10',
    #     'RollingVol_5d', 'Volume_vs_Avg',
    #     # Add any additional engineered features here
    # ]

    # # Evaluation function for one combination
    # def evaluate_combo(feature_list):
    #     try:
    #         indices, actuals, predictions = rolling_window_evaluation(
    #             scaled_df,
    #             feature_cols=feature_list,
    #             target_col='Target',
    #             train_size=60,
    #             test_size=1
    #         )
    #         mae = sklearn.metrics.mean_absolute_error(actuals, predictions)
    #         return (mae, feature_list)
    #     except Exception as e:
    #         return (float('inf'), feature_list)

    # # Try combinations of size 3 to 7
    # combos = []
    # for r in range(3, 8):
    #     combos.extend(list(combinations(candidate_features, r)))

    # # Run in parallel with progress bar
    # print(f"\nRunning evaluation on {len(combos)} combinations...")
    # results = Parallel(n_jobs=-1)(
    #     delayed(evaluate_combo)(list(combo)) for combo in tqdm(combos)
    # )

    # # Sort and display top 10
    # results = [res for res in results if res[0] != float('inf')]
    # results.sort(key=lambda x: x[0])

    # print("\n\U0001f4c8 Top 10 Feature Sets:")
    # for mae, features in results[:10]:
    #     print(f"MAE: {mae:.5f}  \u2192  Features: {features}")
    
    top_feature_sets = [
        ['MA_10d', 'LogVolume', 'Volume_vs_Avg'],
        ['MA_5d', 'LogVolume', 'Volume_vs_Avg'],
        ['LogVolume', 'Returns_1d_x_Close_to_MA5', 'Volume_vs_Avg'],
        ['Close_to_MA5', 'LogVolume', 'Volume_vs_Avg'],
        ['Returns_1d', 'LogVolume', 'Volume_vs_Avg']
    ]

    results = []
    for feature_set in top_feature_sets:
        print(f"\nEvaluating: {feature_set}")
        indices, actuals, predictions = rolling_window_evaluation(
            stock_data,
            feature_cols=feature_set,
            target_col='Target',
            train_size=60,
            test_size=1
        )
        mae = sklearn.metrics.mean_absolute_error(actuals, predictions)
        results.append((mae, indices, actuals, predictions, feature_set))

    # Select best result
    best_result = min(results, key=lambda x: x[0])
    best_mae, best_indices, best_actuals, best_predictions, best_features = best_result

    print(f"\nBest Feature Set: {best_features}")
    print(f"Lowest MAE: {best_mae:.5f}")
    baseline_returns = stock_data.loc[best_indices, 'Returns_1d']
    baseline_mae = sklearn.metrics.mean_absolute_error(best_actuals, baseline_returns)
    print(f"Naive Baseline MAE (Previous Day's Return): {baseline_mae:.5f}")

    # Plot the best result
    plt.figure(figsize=(12, 5))
    plt.plot(best_indices, best_actuals, label="Actual Returns", color="blue")
    plt.plot(best_indices, best_predictions, label="Predicted Returns", color="red")
    plt.title(f"Best Prediction using Features: {', '.join(best_features)}\nMAE: {best_mae:.5f}")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()
   