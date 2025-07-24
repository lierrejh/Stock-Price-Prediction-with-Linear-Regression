# 📈 Stock Price Prediction with Linear Regression

This project applies **linear regression** to forecast the **next-day closing price** of a stock using engineered features such as lagged returns, moving averages, and volume-based indicators.

---

## 🧠 Project Goals

- Use regression techniques to model financial time-series data
- Apply feature engineering for predictive signals
- Evaluate predictions using industry-standard error metrics
- Compare machine learning model performance to a naive baseline
- Interpret model coefficients to understand feature importance

---

## ⚙️ Tools & Libraries

- `yfinance` — stock data fetching
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — regression, evaluation, and scaling
- `matplotlib` — plotting results

---

## 📊 Features Used

Final model used **three key features**:

| Feature           | Description                                        |
|-------------------|----------------------------------------------------|
| `Returns_2d`       | Daily return from two days ago                    |
| `MA_5d`            | 5-day simple moving average of the closing price |
| `Volume_vs_Avg`    | Ratio of today's volume to its 5-day rolling average |

---

## 🧪 Methodology

1. **Download 1-year of daily price & volume data** using `yfinance`
2. **Engineer predictive features** including returns and volume ratios
3. **Create target** as next-day closing price
4. **Split data** into training and test sets (no shuffling)
5. **Standardise features** using `StandardScaler`
6. **Train Linear Regression model**
7. **Evaluate using MAE, MSE**, and compare to a naive baseline
8. **Visualise actual vs predicted prices**

---

## 📈 Results & Findings

### 📌 Final Linear Regression Model:
- Features: `[Returns_2d, MA_5d, Volume_vs_Avg]`
- Coefficients: `[4.54, 15.03, -0.96]`

### 📊 Performance:
| Metric                          | Value     |
|---------------------------------|-----------|
| **Mean Absolute Error (MAE)**   | `2.33`    |
| **Mean Squared Error (MSE)**    | `7.97`    |
| **Naive MAE (5-day MA)**        | `1.79`    |

### ✅ Observations:
- **Moving average (`MA_5d`)** was the most influential feature
- **High relative volume** (`Volume_vs_Avg`) often preceded **price drops**, acting as a contrarian signal
- **Returns_2d** showed weak but consistent predictive power
- **Linear Regression was not able to outperform a simple 5-day MA baseline**

---

## 🧠 Key Takeaways

- Linear models are easy to interpret but **struggle to outperform naive baselines** in stock price prediction
- **Feature selection matters** — removing noisy variables improved generalisation
- Even simple baselines (like a 5-day MA) can **outperform ML models** unless richer features or more complex models are used
- Standardisation improves stability and interpretability of coefficients

---

## 🚀 Future Work

- Try **Ridge Regression** to regularise coefficients
- Use **Random Forests** or **XGBoost** to capture non-linear relationships
- Predict **returns instead of price** for better model stationarity
- Integrate a **trading strategy** and **backtest performance**
