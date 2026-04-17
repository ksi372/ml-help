"""Self-organizing map (MiniSom)
# pip install minisom numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    from minisom import MiniSom
except ImportError:
    raise SystemExit("Install MiniSom: pip install minisom")

USE_SYNTHETIC = True
RNG = np.random.default_rng(26)

if USE_SYNTHETIC:
    n = 180
    age = RNG.integers(20, 71, size=n)
    bmi = RNG.uniform(18, 36, size=n)
    bp = RNG.integers(60, 101, size=n)
    glucose = RNG.integers(80, 191, size=n)
    chol = RNG.integers(150, 281, size=n)
    risk = (
        0.03 * age
        + 0.08 * bmi
        + 0.03 * bp
        + 0.03 * glucose
        + 0.01 * chol
    )
    outcome = (risk > np.median(risk)).astype(int)
    data = pd.DataFrame(
        {
            "Patient_ID": [f"P{i+1}" for i in range(n)],
            "Age": age,
            "BMI": bmi,
            "Blood_Pressure": bp,
            "Glucose": glucose,
            "Cholesterol": chol,
            "Outcome": outcome,
        }
    )
else:
    data = pd.read_csv("som_health_data.csv")

print(data.head())
data = data.dropna().drop_duplicates()
X = data[["Age", "BMI", "Blood_Pressure", "Glucose", "Cholesterol"]].values.astype(float)
X = StandardScaler().fit_transform(X)

som = MiniSom(10, 10, X.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X)
som.train_random(X, 5000)

fig, ax = plt.subplots(figsize=(4, 4))
ax.pcolor(som.distance_map().T, cmap="bone_r")
ax.set_title("SOM neighbor distances (U-matrix)")
plt.tight_layout()
plt.show()
