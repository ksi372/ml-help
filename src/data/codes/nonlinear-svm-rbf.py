"""SVM with RBF kernel
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
RNG = np.random.default_rng(12)

if USE_SYNTHETIC:
    n = 160
    radius = RNG.uniform(10, 18, size=n)
    texture = RNG.uniform(12, 24, size=n)
    smooth = RNG.uniform(0.07, 0.19, size=n)
    symm = RNG.uniform(0.10, 0.28, size=n)
    risk = (radius - 14) ** 2 + (texture - 18) ** 2 + 60 * smooth + 40 * symm
    med = np.median(risk)
    diagnosis = (risk > med).astype(int)
    data = pd.DataFrame(
        {
            "Radius": radius,
            "Texture": texture,
            "Smoothness": smooth,
            "Symmetry": symm,
            "Diagnosis": diagnosis,
        }
    )
else:
    data = pd.read_csv("rbf_svm_tumor.csv")

print(data.head())
data = data.dropna().drop_duplicates()
X = data[["Radius", "Texture", "Smoothness", "Symmetry"]].values
y = data["Diagnosis"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = SVC(kernel="rbf", random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("RBF SVM Confusion Matrix")
plt.show()
