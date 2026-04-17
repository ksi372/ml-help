"""Linear-kernel SVM
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
RNG = np.random.default_rng(11)

if USE_SYNTHETIC:
    n = 140
    cgpa = RNG.uniform(5, 10, size=n)
    apt = RNG.integers(40, 101, size=n)
    code = RNG.integers(35, 101, size=n)
    intern = RNG.integers(0, 2, size=n)
    score = 1.2 * cgpa + 0.03 * apt + 0.035 * code + 0.8 * intern
    placement = (score > 11).astype(int)
    data = pd.DataFrame(
        {
            "CGPA": cgpa,
            "Aptitude_Score": apt,
            "Coding_Score": code,
            "Internship": intern,
            "Placement": placement,
        }
    )
else:
    data = pd.read_csv("linear_svm_placement.csv")

print(data.head())
data = data.dropna().drop_duplicates()
X = data[["CGPA", "Aptitude_Score", "Coding_Score", "Internship"]].values
y = data["Placement"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = SVC(kernel="linear", random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("Linear SVM Confusion Matrix")
plt.show()
