"""Single-layer perceptron (linear classifier)
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
RNG = np.random.default_rng(23)

if USE_SYNTHETIC:
    n = 160
    income = RNG.integers(0, 3, size=n)
    credit = RNG.integers(0, 3, size=n)
    employ = RNG.integers(0, 2, size=n)
    loan_amt = RNG.integers(0, 2, size=n)
    score = 1.5 * income + 1.5 * credit + 1.2 * employ - 1.0 * loan_amt
    status = (score >= 2.5).astype(int)
    data = pd.DataFrame(
        {
            "Income_Level": income,
            "Credit_Score": credit,
            "Employment_Status": employ,
            "Loan_Amount": loan_amt,
            "Loan_Status": status,
        }
    )
else:
    data = pd.read_csv("slp_loan_approval.csv")

print(data.head())
data = data.dropna().drop_duplicates()
X = data[["Income_Level", "Credit_Score", "Employment_Status", "Loan_Amount"]].values
y = data["Loan_Status"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Perceptron(max_iter=1000, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("Single-layer Perceptron Confusion Matrix")
plt.show()
