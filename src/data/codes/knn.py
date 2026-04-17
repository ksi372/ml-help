"""KNN classifier
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
K = 5
RNG = np.random.default_rng(10)

if USE_SYNTHETIC:
    n = 150
    study = RNG.integers(1, 11, size=n)
    att = RNG.integers(50, 101, size=n)
    assign = RNG.integers(40, 101, size=n)
    internal = RNG.integers(10, 51, size=n)
    score = (
        0.8 * study
        + 0.03 * att
        + 0.03 * assign
        + 0.08 * internal
    )
    result = (score > 8.5).astype(int)
    data = pd.DataFrame(
        {
            "Study_Hours": study,
            "Attendance": att,
            "Assignment_Score": assign,
            "Internal_Marks": internal,
            "Result": result,
        }
    )
else:
    data = pd.read_csv("knn_student_data.csv")

print(data.head())
data = data.dropna().drop_duplicates()
X = data[["Study_Hours", "Attendance", "Assignment_Score", "Internal_Marks"]].values
y = data["Result"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"\nK = {K}")
print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

k_values = range(1, 11)
acc_vals = []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    pred = m.predict(X_test)
    acc_vals.append(accuracy_score(y_test, pred))

fig, ax = plt.subplots()
ax.plot(k_values, acc_vals, "o-", linewidth=1.5)
ax.set_xlabel("K value")
ax.set_ylabel("Accuracy")
ax.set_title("Effect of K on KNN Accuracy")
ax.grid(True)

fig2, ax2 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax2)
ax2.set_title("KNN Confusion Matrix")
plt.show()
