import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1':X_pca[:,0],
    'PC2':X_pca[:,1],
    'target':y
})

plt.figure(figsize=(8,6))
for i, target in enumerate(np.unique(y)):
    subset = pca_df[pca_df['target'] == target]
    plt.scatter(subset['PC1'], subset['PC2'], label = target_names[i])

plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("PCA - Iris Data (2D Visualization)")
plt.legend()
plt.grid(True)
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)