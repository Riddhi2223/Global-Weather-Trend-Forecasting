# Global Weather Trend Forecasting

> **PM Accelerator Internship Program** — A complete, end-to-end data science project analyzing global weather data and forecasting temperature trends using machine learning.

---

## Project Overview

This project analyzes the **GlobalWeatherRepository** dataset — a rich collection of daily weather observations across cities worldwide — to:

- Perform comprehensive **exploratory data analysis (EDA)**
- Detect **anomalies** using unsupervised machine learning
- Build **predictive models** to forecast global temperature trends
- Identify the most **influential meteorological features**

The project follows an industry-standard data science workflow: from raw data ingestion through preprocessing, modeling, and evaluation.

---

## Project Structure

```
weather-forecasting/
│
├── data/
│   └── GlobalWeatherRepository.csv    # Raw dataset
│
├── src/
│   ├── preprocessing.py               # Data cleaning & feature engineering
│   ├── eda.py                         # Exploratory data analysis & visualizations
│   ├── models.py                      # ARIMA & XGBoost model training
│   ├── evaluation.py                  # MAE, RMSE metric computation
│   └── advanced.py                    # Anomaly detection & feature importance
│
├── outputs/
│   └── plots/                         # All generated charts & figures
│
├── main.py                            # Entry point — runs the full pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| **Data Manipulation** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn`, `xgboost` |
| **Time Series** | `statsmodels` (ARIMA) |
| **Visualization** | `matplotlib`, `seaborn` |
| **Anomaly Detection** | `sklearn.ensemble.IsolationForest` |

---

## Dataset

- **Source:** `GlobalWeatherRepository.csv`
- **Records:** 130,198 observations
- **Features:** 41 columns

| Category | Features |
|---|---|
| Geographic | `country`, `location_name`, `latitude`, `longitude` |
| Temperature | `temperature_celsius`, `feels_like_celsius` |
| Atmospheric | `pressure_mb`, `humidity`, `cloud`, `uv_index` |
| Wind | `wind_mph`, `wind_kph`, `gust_mph`, `wind_direction` |
| Precipitation | `precip_mm`, `precip_in` |
| Air Quality | `pm2.5`, `pm10`, `carbon_monoxide`, `ozone`, `nitrogen_dioxide` |
| Astronomical | `sunrise`, `sunset`, `moon_phase`, `moon_illumination` |
| Temporal | `last_updated`, `last_updated_epoch` |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/weather-forecasting.git
cd weather-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `GlobalWeatherRepository.csv` inside the `data/` directory.

### 4. Run the pipeline

```bash
python main.py
```

This will execute preprocessing → EDA → modeling → evaluation → advanced analysis in sequence and save all outputs to `outputs/plots/`.

---

## Pipeline Walkthrough

### 1. Data Cleaning (`preprocessing.py`)

| Step | Details |
|---|---|
| DateTime Conversion | Parsed `last_updated` to datetime; dropped unparseable rows |
| Missing Value Imputation | Numeric → median; Categorical → mode |
| Outlier Removal | IQR method on `temperature_celsius` (bounds: −1.75°C to 45.85°C) |
| Feature Engineering | Extracted `year`, `month`, `day`; created `temp_lag_1` lag feature |
| Leakage Prevention | Dropped `temperature_fahrenheit`, `feels_like_*`, and string ID columns |

### 2. Exploratory Data Analysis (`eda.py`)

- **Temperature over time** — Global trend visualization highlighting seasonal variation
- **Precipitation analysis** — Regional rainfall pattern differences
- **Correlation heatmap** — Pearson correlations across all numeric features
- **Temperature vs. Latitude** — Geographic temperature gradient (equator effect)
- **Monthly boxplot** — Seasonal patterns by calendar month
- **Air Quality vs. Temperature** — Environmental impact analysis (r ≈ 0.05)

### 3. Modeling (`models.py`)

See [Models Used](#-models-used) below.

### 4. Evaluation (`evaluation.py`)

See [Evaluation Results](#-evaluation-results) below.

### 5. Advanced Analysis (`advanced.py`)

- **Anomaly Detection** — Isolation Forest (contamination=1%) flagged **1,270** temperature anomalies
- **Feature Importance** — XGBoost gain-based importance ranking of all features

---

## Models Used

### ARIMA (AutoRegressive Integrated Moving Average)
- Classical time-series forecasting model
- Applied to capture autocorrelation and trend components in temperature sequences
- Configured via `statsmodels`

### XGBoost Regressor
- Gradient boosting ensemble method
- Handles non-linear relationships and missing values robustly
- **Configuration:** `n_estimators=100`, chronological train/test split (`shuffle=False`)

---

## Evaluation Results

| Metric | Value | Interpretation |
|---|---|---|
| **MAE** | 3.83°C | Average deviation between predicted and actual temperature |
| **RMSE** | 5.50°C | Penalizes large errors; reflects occasional larger misses |

> These metrics represent strong performance given the global diversity of the dataset spanning all major climate zones.

---

## Key Insights

1. **UV Index is the #1 predictor** (26.5% feature importance) — solar radiation directly drives surface temperature.
2. **Atmospheric pressure** is the 2nd most important feature (19.3%) — pressure systems define weather regimes.
3. **Latitude** confirms geographic impact (11.5%) — temperature decreases with distance from the equator.
4. **Seasonality matters** — monthly patterns show clear temperature cycles across the observation period.
5. **Removing derived features prevents data leakage** — `temperature_fahrenheit` and `feels_like_*` were dropped to ensure honest model evaluation.
6. **Isolation Forest detected 1,270 anomalies** — likely extreme weather events or data quality issues requiring further investigation.

