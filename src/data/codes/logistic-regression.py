"""Logistic regression (L2 / ridge)
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
RNG = np.random.default_rng(6)

if USE_SYNTHETIC:
    n = 160
    age = RNG.integers(20, 71, size=n)
    bmi = RNG.uniform(18, 36, size=n)
    glucose = RNG.integers(75, 201, size=n)
    bp = RNG.integers(60, 101, size=n)
    z = -18 + 0.05 * age + 0.18 * bmi + 0.04 * glucose + 0.03 * bp
    p = 1 / (1 + np.exp(-z))
    diabetes = (RNG.random(n) < p).astype(int)
    data = pd.DataFrame(
        {"Age": age, "BMI": bmi, "Glucose": glucose, "Blood_Pressure": bp, "Diabetes": diabetes}
    )
else:
    data = pd.read_csv("diabetes_data.csv")

print(data.head())
data = data.dropna().drop_duplicates()

X = data[["Age", "BMI", "Glucose", "Blood_Pressure"]].values
y = data["Diabetes"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"\nAccuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("Logistic Regression Confusion Matrix")

fig2, ax2 = plt.subplots()
ax2.scatter(np.arange(len(y_test)), proba, alpha=0.7)
ax2.set_xlabel("Test Sample")
ax2.set_ylabel("Predicted Probability of Diabetes")
ax2.set_title("Predicted Probability")
ax2.grid(True)
plt.show()
