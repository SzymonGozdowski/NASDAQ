import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Wczytanie danych ---
data = pandas.read_csv('15m_data.csv', sep='\t', parse_dates=['DateTime'], dayfirst=False)

# tylko kolumna Close
values = data[['Close']].values.astype(np.float32)

# --- 2. Normalizacja ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values).astype(np.float32)

# --- 3. Funkcja tworząca sekwencje ---
def create_sequences(data, lookback=30, forecast_horizon=1):
    n_samples = len(data) - lookback - forecast_horizon + 1
    X = np.zeros((n_samples, lookback, data.shape[1]), dtype=np.float32)
    y = np.zeros((n_samples, forecast_horizon, data.shape[1]), dtype=np.float32)
    for i in range(n_samples):
        X[i] = data[i:i+lookback]
        y[i] = data[i+lookback:i+lookback+forecast_horizon]
    return X, y

# --- 4. Przygotowanie danych ---
lookback = 30
forecast_horizon = 1
X, y = create_sequences(scaled, lookback, forecast_horizon)

# --- 5. Podział na train/val/test ---
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# wybór kolumny Close (tu tylko jedna kolumna, index 0)
target_col = 0
y_train = y_train[:, -1, target_col]
y_val = y_val[:, -1, target_col]
y_test = y_test[:, -1, target_col]

# --- 6. Reshape dla modelu ---
n_train, n_steps, n_feats = X_train.shape
n_val = X_val.shape[0]
n_test = X_test.shape[0]

X_train = X_train.reshape(n_train, n_steps * n_feats)
X_val = X_val.reshape(n_val, n_steps * n_feats)
X_test = X_test.reshape(n_test, n_steps * n_feats)

# --- 7. Model Random Forest ---
rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# --- 8. Predykcje ---
rf_y_pred = rf.predict(X_test)

# --- 9. Metryki ---
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print(f"RF: MSE: {rf_mse:.6f}, MAE: {rf_mae:.6f}, R²: {rf_r2:.4f}")

# --- 10. Wykres ---
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual")
plt.plot(rf_y_pred, label="Predicted Random Forest")
plt.legend()
plt.title("NASDAQ Close Price Prediction (Test Set)")
plt.show()
