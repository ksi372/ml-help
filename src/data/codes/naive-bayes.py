"""Naive Bayes (GaussianNB for mixed numeric features)
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
RNG = np.random.default_rng(3)

if USE_SYNTHETIC:
    n = 150
    promo = RNG.integers(0, 6, size=n)
    susp = RNG.integers(0, 2, size=n)
    msg_len = RNG.integers(40, 251, size=n)
    url_c = RNG.integers(0, 5, size=n)
    cat = []
    for i in range(n):
        if promo[i] >= 3 and susp[i] == 1 and url_c[i] >= 2:
            cat.append("Spam")
        elif promo[i] >= 2 and msg_len[i] > 120:
            cat.append("Promotional")
        else:
            cat.append("Important")
    data = pd.DataFrame(
        {
            "Promo_keywords": promo,
            "Suspicious_words": susp,
            "Message_length": msg_len,
            "URL_count": url_c,
            "Category": cat,
        }
    )
else:
    data = pd.read_csv("email_classification_data.csv")

print(data.head())

data = data.dropna().drop_duplicates()
X = data[["Promo_keywords", "Suspicious_words", "Message_length", "URL_count"]].values
y = data["Category"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
acc = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"\nAccuracy = {acc:.4f}")
print("\n", classification_report(y_test, y_pred, zero_division=0))

from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("Naive Bayes Confusion Matrix")
plt.show()
