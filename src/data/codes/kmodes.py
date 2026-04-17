"""K-modes (categorical clustering) — manual matching distance
# pip install numpy pandas
"""
import numpy as np
import pandas as pd

USE_SYNTHETIC = True
K = 3
MAX_ITER = 20
RNG = np.random.default_rng(14)

if USE_SYNTHETIC:
    data = pd.DataFrame(
        {
            "Gender": ["Male", "Female"] * 5,
            "Branch": ["CSE", "IT", "ECE"] * 3 + ["CSE", "IT"],
            "Residence": ["Hostel", "Day"] * 5,
            "Study_Style": ["Night", "Morning"] * 5,
            "Attendance_Level": ["High", "Medium", "Low"] * 3 + ["High", "Medium"],
            "Result_Status": ["Pass", "Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail", "Pass"],
        }
    )
else:
    data = pd.read_csv("kmodes_student_profiles.csv")

print(data)
data = data.dropna().drop_duplicates()
Xstr = data.astype(str).values
n, m = Xstr.shape

rand_idx = RNG.permutation(n)[:K]
modes = Xstr[rand_idx].copy()
cluster_idx = np.zeros(n, dtype=int)

for _iter in range(MAX_ITER):
    old = cluster_idx.copy()
    for i in range(n):
        dist = np.zeros(K)
        for k in range(K):
            dist[k] = np.sum(Xstr[i] != modes[k])
        cluster_idx[i] = int(np.argmin(dist))
    for k in range(K):
        mask = cluster_idx == k
        if not np.any(mask):
            continue
        cluster_pts = Xstr[mask]
        for j in range(m):
            vals, counts = np.unique(cluster_pts[:, j], return_counts=True)
            modes[k, j] = vals[np.argmax(counts)]
    if np.array_equal(old, cluster_idx):
        break

cost = sum(np.sum(Xstr[i] != modes[cluster_idx[i]]) for i in range(n))
print(f"\nTotal clustering cost = {cost}")
print("Cluster assignments:", cluster_idx)
print("Modes:\n", modes)
