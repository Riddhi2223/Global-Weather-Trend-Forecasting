from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

def arima_model(series):
    model = ARIMA(series, order=(5,1,0))
    return model.fit()

def xgboost_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model