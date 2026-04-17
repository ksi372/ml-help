"""
Linear regression — numpy / sklearn
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Settings
USE_SYNTHETIC = True
CSV_FILE = "bodyfat_data.csv"
RNG = np.random.default_rng(1)

if USE_SYNTHETIC:
    n = 100
    age = RNG.integers(18, 61, size=n)
    daily_calorie = RNG.integers(1800, 3501, size=n)
    sleep_hours = RNG.uniform(5, 9, size=n)
    workout_sessions = RNG.integers(0, 7, size=n)
    body_fat = (
        0.25 * age
        + 0.004 * daily_calorie
        - 1.8 * sleep_hours
        - 1.5 * workout_sessions
        + RNG.normal(0, 2, size=n)
        + 8
    )
    data = pd.DataFrame(
        {
            "Age": age,
            "DailyCalorieIntake": daily_calorie,
            "SleepHours": sleep_hours,
            "WorkoutSessions": workout_sessions,
            "BodyFat": body_fat,
        }
    )
else:
    data = pd.read_csv(CSV_FILE)

print(data.head())

data = data.dropna().drop_duplicates()
X = data[["Age", "DailyCalorieIntake", "SleepHours", "WorkoutSessions"]].values
y = data["BodyFat"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
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
ax.set_xlabel("Actual Body Fat")
ax.set_ylabel("Predicted Body Fat")
ax.set_title("Actual vs Predicted Body Fat")
ax.grid(True)

fig2, ax2 = plt.subplots()
idx = np.arange(len(y_test))
ax2.plot(idx, y_test, "b-o", label="Actual", linewidth=1.5)
ax2.plot(idx, y_pred, "r-*", label="Predicted", linewidth=1.5)
ax2.legend()
ax2.set_title("Actual and Predicted Values")
ax2.set_xlabel("Test Sample")
ax2.set_ylabel("Body Fat")
ax2.grid(True)
plt.show()
