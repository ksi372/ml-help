"""Multi-layer perceptron
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
RNG = np.random.default_rng(24)

if USE_SYNTHETIC:
    n = 180
    age = RNG.integers(0, 3, size=n)
    bmi = RNG.integers(0, 3, size=n)
    phys = RNG.integers(0, 3, size=n)
    bp = RNG.integers(0, 3, size=n)
    fam = RNG.integers(0, 2, size=n)
    risk = (
        0.7 * age
        + 1.0 * bmi
        - 0.8 * phys
        + 0.9 * bp
        + 1.2 * fam
    )
    risk_level = (risk >= 2.5).astype(int)
    data = pd.DataFrame(
        {
            "Age_Group": age,
            "BMI_Category": bmi,
            "Physical_Activity": phys,
            "Blood_Pressure": bp,
            "Family_History": fam,
            "Risk_Level": risk_level,
        }
    )
else:
    data = pd.read_csv("mlp_disease_risk.csv")

print(data.head())
data = data.dropna().drop_duplicates()
cols = ["Age_Group", "BMI_Category", "Physical_Activity", "Blood_Pressure", "Family_History"]
X = data[cols].values
y = data["Risk_Level"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score  = {f1_score(y_test, y_pred, zero_division=0):.4f}")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("MLP Confusion Matrix")
plt.show()
