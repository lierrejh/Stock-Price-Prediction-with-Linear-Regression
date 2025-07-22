# 📈 Stock Price Prediction with Linear Regression

This project uses a simple **Linear Regression model** to predict the **next-day closing price** of a stock based on engineered features such as lagged returns, moving averages, and volume.

---

## 🚀 Project Goals

- Apply linear regression to financial time series data
- Learn and implement feature engineering techniques
- Evaluate predictions using MAE and MSE
- Visualise actual vs predicted prices

---

## 📊 Features Engineered

- **Lagged Returns**: Daily percentage returns from 1 and 2 days ago
- **Moving Averages**: Rolling average of previous N days’ prices
- **Volume**: Raw or transformed daily trading volume

---

## 🛠️ Tools & Libraries

- [`yfinance`](https://pypi.org/project/yfinance/) – to fetch stock data
- `pandas`, `numpy` – for data manipulation
- `scikit-learn` – for Linear Regression and evaluation metrics
- `matplotlib` – for visualisation

---

## ⚙️ How It Works

1. **Fetch Data**: Load 1 year of daily stock data using `yfinance`
2. **Create Features**: Compute lagged returns, moving averages, and volume
3. **Target Setup**: Define the target as the next day's closing price
4. **Train/Test Split**: Split the dataset into training and test sets
5. **Train Model**: Use `LinearRegression` from `scikit-learn`
6. **Evaluate**: Use MAE and MSE on the test set
7. **Visualise**: Plot predictions vs actual closing prices
