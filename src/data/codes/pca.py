"""Principal component analysis
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

USE_SYNTHETIC = True
RNG = np.random.default_rng(25)

if USE_SYNTHETIC:
    n = 200
    time_on = RNG.uniform(5, 25, size=n)
    pages = np.round(time_on + RNG.normal(0, 2, n) + 3).astype(int)
    cart = np.round(0.2 * pages + RNG.normal(0, 1, n)).astype(int)
    purchase = np.round(0.6 * cart + RNG.normal(0, 1, n) + 1).astype(int)
    order_val = 100 + 80 * cart.astype(float) + RNG.normal(0, 50, size=n)
    discount = 70 - 0.15 * order_val + RNG.normal(0, 5, size=n)
    returns = 5 + 0.1 * discount + RNG.normal(0, 2, size=n)
    reviews = np.round(0.8 * purchase + RNG.normal(0, 1, n)).astype(int)
    data = pd.DataFrame(
        {
            "Time_On_Site": time_on,
            "Pages_Viewed": pages,
            "Cart_Additions": cart,
            "Purchase_Frequency": purchase,
            "Avg_Order_Value": order_val,
            "Discount_Usage": discount,
            "Return_Rate": returns,
            "Review_Count": reviews,
        }
    )
else:
    data = pd.read_csv("pca_customer_behavior.csv")

print(data.head())
data = data.dropna().drop_duplicates()
X = data.values.astype(float)
X = StandardScaler().fit_transform(X)

pca = PCA()
score = pca.fit_transform(X)
explained_ratio = pca.explained_variance_ratio_ * 100

print("Explained variance (% per component):")
print(explained_ratio)

fig, ax = plt.subplots()
ax.bar(range(1, len(explained_ratio) + 1), explained_ratio)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax.set_title("PCA Explained Variance")

fig2, ax2 = plt.subplots()
ax2.scatter(score[:, 0], score[:, 1], alpha=0.6)
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title("PCA Projection on First Two Components")
ax2.grid(True)

print("PCA Components (rows = features, cols = components):\n", pca.components_.T)
plt.show()
