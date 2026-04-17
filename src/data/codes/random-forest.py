"""Random Forest
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
N_TREES = 100
RNG = np.random.default_rng(15)

if USE_SYNTHETIC:
    n = 180
    income = RNG.uniform(2, 10, size=n)
    credit = RNG.integers(450, 851, size=n)
    emp = RNG.integers(0, 16, size=n)
    loan_amt = RNG.uniform(2, 12, size=n)
    existing = RNG.integers(0, 4, size=n)
    score = (
        0.7 * income
        + 0.01 * credit
        + 0.15 * emp
        - 0.5 * loan_amt
        - 0.8 * existing
    )
    status = (score > 7.5).astype(int)
    data = pd.DataFrame(
        {
            "Annual_Income": income,
            "Credit_Score": credit,
            "Employment_Years": emp,
            "Loan_Amount": loan_amt,
            "Existing_Loans": existing,
            "Loan_Status": status,
        }
    )
else:
    data = pd.read_csv("random_forest_loan.csv")

print(data.head())
data = data.dropna().drop_duplicates()
cols = [
    "Annual_Income",
    "Credit_Score",
    "Employment_Years",
    "Loan_Amount",
    "Existing_Loans",
]
X = data[cols].values
y = data["Loan_Status"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=N_TREES, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\nN Trees = {N_TREES}")
print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

tree_vals = [10, 20, 50, 100, 150]
acc_vals = []
for nt in tree_vals:
    m = RandomForestClassifier(n_estimators=nt, random_state=42).fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_vals.append(accuracy_score(y_test, pred))

fig, ax = plt.subplots()
ax.plot(tree_vals, acc_vals, "o-", linewidth=1.5)
ax.set_xlabel("Number of Trees")
ax.set_ylabel("Accuracy")
ax.set_title("Effect of Number of Trees on Random Forest Accuracy")
ax.grid(True)

fig2, ax2 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax2)
ax2.set_title("Random Forest Confusion Matrix")
plt.show()
