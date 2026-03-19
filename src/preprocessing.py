import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    return df

def clean_data(df):
    df = df.drop_duplicates()

    # Convert datetime
    df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
    df = df.dropna(subset=['last_updated'])

    # Fill missing values (FIXED: no inplace warning)
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def remove_outliers(df):
    Q1 = df['temperature_celsius'].quantile(0.25)
    Q3 = df['temperature_celsius'].quantile(0.75)
    IQR = Q3 - Q1

    df = df[(df['temperature_celsius'] >= Q1 - 1.5 * IQR) &
            (df['temperature_celsius'] <= Q3 + 1.5 * IQR)]

    return df

def feature_engineering(df):
    df = df.sort_values('last_updated')

    df['year'] = df['last_updated'].dt.year
    df['month'] = df['last_updated'].dt.month
    df['day'] = df['last_updated'].dt.day

    df['temp_lag_1'] = df['temperature_celsius'].shift(1)

    df = df.dropna()

    return df

def drop_unused_columns(df):
    return df.drop(columns=[
        'country','location_name','timezone','condition_text',
        'wind_direction','sunrise','sunset','moonrise','moonset','moon_phase',

        # Remove leakage features
        'temperature_fahrenheit',
        'feels_like_celsius',
        'feels_like_fahrenheit'
    ], errors='ignore')