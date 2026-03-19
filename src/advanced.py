from sklearn.ensemble import IsolationForest

def anomaly_detection(df):
    model = IsolationForest(contamination=0.01)
    df['anomaly'] = model.fit_predict(df[['temperature_celsius']])
    return df