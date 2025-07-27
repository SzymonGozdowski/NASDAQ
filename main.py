import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pandas.read_csv('15m_data.csv', sep='\t', parse_dates=['DateTime'], dayfirst=False)
features = ["Open", "High", "Low", "Close", "Volume", "TickVolume"]
values = data[features].values
# print(data['High'])

# --- 1. Normalize the data ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)


# --- 2. Function to create sliding windows ---
def create_sequences(data, lookback=30, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+forecast_horizon])
    return np.array(X), np.array(y)


# Use last 30 days to predict next 1 day
lookback = 30
forecast_horizon = 1
X, y = create_sequences(scaled, lookback, forecast_horizon)

# --- 3. Train/validation/test split (no shuffle!) ---
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# --- 4. Shapes ---
print("X_train:", X_train.shape)  # (samples, lookback, 1)
print("y_train:", y_train.shape)  # (samples, forecast_horizon, 1)


plt.figure(1)
plt.plot(range(len(data)), data['High'])
plt.show()
