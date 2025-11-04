# 📘 Project 73 - UMAP Visualization for High-Dimensional Data

import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load high-dimensional dataset
digits = load_digits()
X = digits.data
y = digits.target
print("Original Data Shape:", X.shape)

# 2️⃣ Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Apply UMAP (reduce to 2D)
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 4️⃣ Plot UMAP visualization
plt.figure(figsize=(10,7))
scatter = plt.scatter(X_umap[:,0], X_umap[:,1], c=y, cmap='tab10', s=20)
plt.colorbar(scatter, label='Digit Label')
plt.title("UMAP Visualization of Handwritten Digits Dataset")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.show()
