import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
df_pca['Target'] = y

plt.figure(figsize=(8,6))
for target, color in zip([0,1,2], ['red','green', 'blue']):
    subset = df_pca[df_pca['Target'] == target]
    plt.scatter(subset['PC1'], subset['PC2'], label = iris.target_names[target], color = color)

plt.title("PCA - 2D Projection of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)

