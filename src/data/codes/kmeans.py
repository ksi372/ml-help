"""K-means + silhouette
# pip install numpy pandas scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

USE_SYNTHETIC = True
K = 3
RNG = np.random.default_rng(13)

if USE_SYNTHETIC:
    n = 150
    data = pd.DataFrame(
        {
            "CGPA": RNG.uniform(5, 10, size=n),
            "Aptitude_Score": RNG.integers(40, 101, size=n),
            "Coding_Score": RNG.integers(35, 101, size=n),
            "Attendance": RNG.integers(55, 101, size=n),
        }
    )
else:
    data = pd.read_csv("kmeans_student_grouping.csv")

print(data.head())
data = data.dropna().drop_duplicates()
X = data.values.astype(float)
X = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=K, n_init=5, random_state=42)
labels = km.fit_predict(X)
inertia = km.inertia_
print(f"Inertia = {inertia:.4f}")

s = silhouette_samples(X, labels)
sil_avg = silhouette_score(X, labels)
print(f"Silhouette Score = {sil_avg:.4f}")

fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7)
ax.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    c="k",
    marker="x",
    s=200,
    linewidths=3,
    label="centroids",
)
ax.set_xlabel("Normalized CGPA")
ax.set_ylabel("Normalized Aptitude Score")
ax.set_title("K-Means Clusters")
ax.grid(True)
plt.show()
