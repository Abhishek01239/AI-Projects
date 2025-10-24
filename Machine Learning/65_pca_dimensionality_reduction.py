import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Original Shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y

plt.figure(figsize=(8,6))
for label in pca_df['Target'].unique():
    subset = pca_df[pca_df['Target'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], label=data.target_names[label])

plt.title("PCA - 2D Projection of Iris Dataset", fontsize=14, fontweight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
