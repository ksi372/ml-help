"""AdaBoost (SAMME = discrete AdaBoost multiclass)
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
N_ESTIMATORS = 50
RNG = np.random.default_rng(21)

if USE_SYNTHETIC:
    n = 180
    speed = RNG.integers(60, 141, size=n)
    brakes = RNG.integers(0, 21, size=n)
    night = RNG.integers(0, 26, size=n)
    mobile = RNG.integers(0, 61, size=n)
    acc_hist = RNG.integers(0, 5, size=n)
    risk_score = (
        0.03 * speed
        + 0.18 * brakes
        + 0.12 * night
        + 0.08 * mobile
        + 0.9 * acc_hist
    )
    risk_level = (risk_score > np.median(risk_score)).astype(int)
    data = pd.DataFrame(
        {
            "Avg_Speed": speed,
            "Harsh_Brakes": brakes,
            "Night_Driving_Hours": night,
            "Mobile_Usage": mobile,
            "Accident_History": acc_hist,
            "Risk_Level": risk_level,
        }
    )
else:
    data = pd.read_csv("adaboost_driver_risk.csv")

print(data.head())
data = data.dropna().drop_duplicates()
cols = ["Avg_Speed", "Harsh_Brakes", "Night_Driving_Hours", "Mobile_Usage", "Accident_History"]
X = data[cols].values
y = data["Risk_Level"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

base = DecisionTreeClassifier(max_depth=1, random_state=42)
model = AdaBoostClassifier(
    estimator=base, n_estimators=N_ESTIMATORS, random_state=42
).fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("AdaBoost Confusion Matrix")
plt.show()
