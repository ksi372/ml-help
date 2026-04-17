"""Decision tree — deviance ≈ log_loss criterion
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
RNG = np.random.default_rng(4)

if USE_SYNTHETIC:
    n = 120
    att = RNG.integers(0, 3, size=n)
    internal = RNG.integers(0, 3, size=n)
    assign = RNG.integers(0, 2, size=n)
    study = RNG.integers(0, 3, size=n)
    res = []
    for i in range(n):
        s = att[i] + internal[i] + assign[i] + study[i]
        res.append("Pass" if s >= 4 else "Fail")
    data = pd.DataFrame(
        {
            "Attendance": att,
            "Internal_Marks": internal,
            "Assignment_Status": assign,
            "Study_Hours": study,
            "Result": res,
        }
    )
else:
    data = pd.read_csv("student_performance_data.csv")

print(data.head())
data = data.dropna().drop_duplicates()

X = data[["Attendance", "Internal_Marks", "Assignment_Status", "Study_Hours"]].values
y = data["Result"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = DecisionTreeClassifier(criterion="log_loss", random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"\nAccuracy = {accuracy_score(y_test, y_pred):.4f}")

for lab in np.unique(y):
    m = precision_recall_fscore_support(
        y_test == lab, y_pred == lab, average="binary", zero_division=0
    )
    print(f"\nClass: {lab}")
    print(f"Precision = {m[0]:.4f}, Recall = {m[1]:.4f}, F1 = {m[2]:.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("Decision Tree (log_loss) Confusion Matrix")
plt.show()
