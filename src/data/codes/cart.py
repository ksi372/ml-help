"""CART — Gini impurity
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
RNG = np.random.default_rng(5)

if USE_SYNTHETIC:
    n = 180
    amt = RNG.integers(500, 80001, size=n)
    per_day = RNG.integers(1, 9, size=n)
    loc_m = RNG.integers(0, 2, size=n)
    dev = RNG.integers(0, 2, size=n)
    ttime = RNG.integers(0, 2, size=n)
    fraud = []
    for i in range(n):
        risk = sum(
            [amt[i] > 30000, per_day[i] > 4, loc_m[i] == 1, dev[i] == 1, ttime[i] == 1]
        )
        fraud.append("Fraud" if risk >= 3 else "Legitimate")
    data = pd.DataFrame(
        {
            "Transaction_amount": amt,
            "Transactions_per_day": per_day,
            "Location_mismatch": loc_m,
            "Device_change": dev,
            "Transaction_time": ttime,
            "Fraud_Status": fraud,
        }
    )
else:
    data = pd.read_csv("fraud_detection_data.csv")

print(data.head())
data = data.dropna().drop_duplicates()

X = data[
    [
        "Transaction_amount",
        "Transactions_per_day",
        "Location_mismatch",
        "Device_change",
        "Transaction_time",
    ]
].values
y = data["Fraud_Status"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = DecisionTreeClassifier(criterion="gini", random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nAccuracy = {accuracy_score(y_test, y_pred):.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("CART (Gini) Confusion Matrix")
plt.show()
