# Stock Return Prediction with Linear Regression

This project explores using linear regression to predict **next-day stock returns** using features derived from historical price and volume data. The goal is to balance simplicity and insight — staying within the bounds of linear models while iterating on features, evaluation methods, and prediction targets to maximise accuracy and robustness.

---

## 📌 Objective

Build a linear model that predicts **next-day stock returns**, compare its performance to a naive benchmark (e.g. return = previous return), and identify the most effective features and validation strategies for this task.

---

## 🔧 Process Overview

### 1. Data Acquisition

* Pulled 1 year of daily OHLCV data using `yfinance`
* Primary columns used: `Close` and `Volume`

### 2. Feature Engineering

Constructed several predictive variables:

* **Lagged Returns:**

  * `Returns_1d`, `Returns_2d`, `Returns_5d` — raw percentage changes over past days
* **Moving Averages:**

  * `MA_5d`, `MA_10d` — simple moving averages to detect short-term trend direction
* **Ratios & Divergence Indicators:**

  * `Close_to_MA5` — detects how stretched price is above or below short-term average
  * `MA_ratio` — compares fast vs slow MAs for crossover-like signals
* **Volume Dynamics:**

  * `LogVolume` — stabilises volume scale
  * `Volume_vs_Avg`, `RollingVol_5d` — indicates volume surges relative to past norms
* **Interaction Terms (Non-linear Relations Within Linear Models):**

  * `Returns_1d_x_Close_to_MA5`, `Returns_2d_x_MA10` — useful for capturing signal when directional move aligns with technical divergence

### 3. Target Variable

* Originally tried to predict **next-day price** directly → very poor results with linear models
* Switched to predicting **next-day return**, which:

  * Is mean-stationary (does not trend upward forever like price)
  * Has a narrower range (\~-10% to +10%)
  * Is more suitable for linear extrapolation

✅ **Conclusion:** Returns are a more appropriate target than price when using linear models

---

## ⚖️ Feature Scaling

* All features were scaled using `StandardScaler`
* This:

  * Prevents magnitude-dominant features (like volume) from dominating coefficients
  * Helps linear regression converge more stably
  * Allows fairer feature comparison

---

## 🧪 Evaluation Methods

### ❌ Train-Test Split (Initial Attempt)

* Used `train_test_split` with `shuffle=False`
* Problems:

  * Model fits in-sample data well but fails to generalise
  * Lower realism: doesn’t simulate real-time performance
  * Higher MAEs observed consistently

### ✅ Rolling Window Cross-Validation (Final Method)

* Rolling 60-day training windows with 1-day test predictions
* Simulates a more realistic out-of-sample, walk-forward prediction environment
* Averaged MAE across all test windows

**Result:** Rolling window yielded **more stable and conservative error metrics**, and reliably identified meaningful features

---

## 📉 Model Performance

### Prediction Characteristics

* Linear models predict conservative, near-zero returns most of the time
* Significant moves in prediction only happen when inputs strongly deviate (e.g., price far from MA + spike in volume)
* Cannot capture sudden nonlinear jumps or regime changes

### Best Model on AAPL (Using Rolling Evaluation)

* **Features:** `['MA_10d', 'LogVolume', 'Volume_vs_Avg']`
* **Rolling Window MAE:** `0.01492`
* **Naive Baseline MAE:** `0.69331`

This means the model outperforms the naive assumption of "tomorrow’s return = today’s" by a factor of \~46x

---

## 📊 Key Findings and Lessons Learned

### 1. **Returns are better targets than prices**

* Prices are trending and non-stationary — linear models underperform heavily when predicting future price
* Returns are roughly mean-reverting and zero-centred — a better fit for linear assumptions

### 2. **Train/Test Split ≠ Time-Series Best Practice**

* Rolling validation respects temporal ordering and avoids data leakage
* Performance measured this way is **much closer to live trading expectation**

### 3. **Feature Combinations Matter**

* Not all features add value
* Top-performing combinations included 3–4 features max
* Redundant or weakly correlated inputs degrade generalisation

### 4. **Volume Features Were Highly Predictive**

* Volume spikes relative to rolling average (e.g., `Volume_vs_Avg`) often correlated with significant return signals
* Log-transformed volume reduced noise and improved learning

### 5. **Interaction Terms Enhance Linear Expressiveness**

* Multiplying two meaningful features (e.g., `Returns_1d * Close_to_MA5`) creates more nuanced patterns
* Still keeps the model strictly linear — great for interpretability

### 6. **Predictions Will Always Be Conservative**

* Model outputs mostly cluster around 0
* Does not mean it’s useless — sharp signals can still be extracted
* Good for directional bias or filtering high-risk environments

### 7. **Baseline Comparison Is Critical**

* Always evaluate against naive strategy (e.g., previous return)
* High performance must beat naive MAE — otherwise, your model adds no value

---

## 📈 Summary of Best Models

| Rank | MAE     | Features                                                            |
| ---- | ------- | ------------------------------------------------------------------- |
| 1    | 0.00877 | \['MA\_10d', 'LogVolume', 'Volume\_vs\_Avg']                        |
| 2    | 0.00886 | \['MA\_5d', 'LogVolume', 'Volume\_vs\_Avg']                         |
| 3    | 0.00889 | \['LogVolume', 'Returns\_1d\_x\_Close\_to\_MA5', 'Volume\_vs\_Avg'] |
| 4    | 0.00889 | \['Close\_to\_MA5', 'LogVolume', 'Volume\_vs\_Avg']                 |
| 5    | 0.00890 | \['Returns\_1d', 'LogVolume', 'Volume\_vs\_Avg']                    |

---

## 🧭 Next Steps / Extensions

* Add regularisation (Ridge, Lasso) to stabilise coefficients
* Build binary classifier: will return > 0?
* Use cumulative return signal to construct a trading strategy
* Ensemble top 3 models for smoother prediction
* Try PCA or feature selection algorithms

---

## 🏁 Final Word

This project demonstrates that with well-thought-out features, realistic validation, and domain-specific insight, even a **simple linear model** can extract meaningful signals from financial data — especially when predicting **returns**, not **prices**.
