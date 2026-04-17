"""Gradient boosting regression (sklearn — analogous to LS boosting workflow)
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
N_ESTIMATORS = 100
RNG = np.random.default_rng(22)

if USE_SYNTHETIC:
    n = 250
    rainfall = RNG.integers(650, 1101, size=n)
    temp = RNG.uniform(22, 32, size=n)
    ph = RNG.uniform(5.5, 7.3, size=n)
    fert = RNG.integers(80, 181, size=n)
    pest = RNG.integers(8, 26, size=n)
    irrig = RNG.integers(5, 19, size=n)
    hum = RNG.integers(50, 86, size=n)
    yield_ = (
        0.0025 * rainfall
        - 0.08 * temp
        + 0.9 * ph
        + 0.012 * fert
        - 0.015 * pest
        + 0.05 * irrig
        + 0.01 * hum
        + RNG.normal(0, 0.2, size=n)
    )
    data = pd.DataFrame(
        {
            "Rainfall_mm": rainfall,
            "Avg_Temperature": temp,
            "Soil_pH": ph,
            "Fertilizer_Used_kg": fert,
            "Pesticide_Used_kg": pest,
            "Irrigation_Hours": irrig,
            "Humidity": hum,
            "Crop_Yield": yield_,
        }
    )
else:
    data = pd.read_csv("xgboost_crop_yield.csv")

print(data.head())
data = data.dropna().drop_duplicates()
feat = [
    "Rainfall_mm",
    "Avg_Temperature",
    "Soil_pH",
    "Fertilizer_Used_kg",
    "Pesticide_Used_kg",
    "Irrigation_Hours",
    "Humidity",
]
X = data[feat].values
y = data["Crop_Yield"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=N_ESTIMATORS, max_depth=10, learning_rate=0.1, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R^2  = {r2:.4f}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.set_xlabel("Actual Yield")
ax.set_ylabel("Predicted Yield")
ax.set_title("Actual vs Predicted Crop Yield")
ax.grid(True)
plt.show()
