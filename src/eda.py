import matplotlib.pyplot as plt
import seaborn as sns

def plot_temperature(df):
    plt.figure()
    plt.plot(df['last_updated'], df['temperature_celsius'])
    plt.title("Temperature Trend")
    plt.savefig("outputs/plots/temp_trend.png")

def plot_precipitation(df):
    plt.figure()
    plt.plot(df['last_updated'], df['precip_mm'])
    plt.title("Precipitation Trend")
    plt.savefig("outputs/plots/precipitation.png")

def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/plots/correlation.png")