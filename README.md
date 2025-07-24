# Stock Return Prediction with Linear Regression

This project uses linear regression to predict next-day returns for a given stock using engineered features derived from historical prices and volume data. The workflow explores different feature combinations and evaluation techniques to identify the most predictive input set for the linear model.

---

## 📌 Objective

Predict next-day returns using only linear models and evaluate them against a naive baseline. Optimise accuracy using feature engineering, standardisation, and rolling-window validation.

---

## 🔧 Process Overview

### 1. Data Acquisition

* Stock data is fetched using `yfinance`.
* Focuses on daily closing prices and trading volumes.

### 2. Feature Engineering

Includes:

* Lagged returns (`Returns_1d`, `Returns_2d`, `Returns_5d`)
* Moving averages (`MA_5d`, `MA_10d`)
* Price-to-average ratios (`Close_to_MA5`, `MA_ratio`)
* Volume indicators (`LogVolume`, `Volume_vs_Avg`, `RollingVol_5d`)
* Interaction terms (`Returns_1d * Close_to_MA5`, `Returns_2d * MA_10d`)

### 3. Target Variable

* Next-day return: `pct_change().shift(-1)` on closing prices.

### 4. Data Scaling

* Standardised all features using `StandardScaler`.
* This significantly improves model convergence and stabilises predictions.

---

## 🧪 Model Evaluation

### Train-Test Split

* Initially used `train_test_split()` (80/20, no shuffle).
* Result: Higher MAE and overly optimistic in-sample fit.

### Rolling Window Cross-Validation ✅

* Uses a sliding window of 60 days training, 1 day testing.
* Repeats over the entire dataset.
* **This method proved far more realistic and robust** for time-series prediction.

### Baseline Comparison

* Naive baseline predicts return(t) = return(t-1).
* MAE of naive baseline was \~0.69 (AAPL), while best linear model achieved \~0.015 — a **46x improvement**.

### Model Behaviour

* Predictions mostly centre around 0% (mean return), with subtle adjustments during high volatility.
* Linear models naturally dampen extreme predictions.
* Best performance came from subtle signals rather than large return swings.

---

## ✅ Best Observed Result (AAPL)

* **Feature Set:** `['MA_10d', 'LogVolume', 'Volume_vs_Avg']`
* **Rolling Window MAE:** `0.01492`
* **Baseline MAE:** `0.69331`

---

## 📊 Lessons Learned

* Train/test splits are not ideal for time series.
* Rolling-window evaluation gives realistic out-of-sample errors.
* Simple features like volume and moving average ratios outperform most lagged return terms.
* Feature standardisation is essential for linear models.
* Linear regression is conservative — captures signal, but rarely predicts extremes.
* Adding interaction terms improves model expressiveness without breaking linear constraints.

