from src.preprocessing import *
from src.eda import *
from src.models import *
from src.evaluation import *
from src.advanced import *

from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = load_data("data/GlobalWeatherRepository.csv")

# Preprocessing
df = clean_data(df)
df = remove_outliers(df)
df = feature_engineering(df)
df = drop_unused_columns(df)

# EDA
plot_temperature(df)
plot_precipitation(df)
correlation_heatmap(df)

# Modeling
X = df.drop(columns=['temperature_celsius','last_updated'])
y = df['temperature_celsius']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

model = xgboost_model(X_train, y_train)
preds = model.predict(X_test)

# Evaluation
results = evaluate(y_test, preds)
print("\nModel Results:", results)

# Feature Importance
importance = model.feature_importances_

feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nTop Features:\n", feat_imp.head(10))

# Anomaly Detection
df = anomaly_detection(df)

print("\nProject executed successfully!")