import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import seaborn as sns

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

# 4. Flatten X and select 'Close' as target
target_col = 3  # index of 'Close'
y_train = y_train[:, -1, target_col]
y_val = y_val[:, -1, target_col]
y_test = y_test[:, -1, target_col]

# Get shapes per split
n_train, n_steps, n_feats = X_train.shape
n_val = X_val.shape[0]
n_test = X_test.shape[0]

# Reshape for 2D input
X_train = X_train.reshape(n_train, n_steps * n_feats)
X_val = X_val.reshape(n_val, n_steps * n_feats)
X_test = X_test.reshape(n_test, n_steps * n_feats)



# train models
# model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
# max_depth=1, random_state=0).fit(X_train, y_train)
# model_gbr.score(X_test, y_test)

neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(X_train, y_train)

# 6. Predictions
y_pred = neigh.predict(X_test)

# 7. Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")

# 8. Plot prediction vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("NASDAQ Close Price Prediction (Test Set)")
plt.show()
# plt.figure(1)
# plt.plot(range(len(data)), data['High'])
# plt.show()
