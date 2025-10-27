import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import pandas as pd

X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kpca = KernelPCA(n_components=2, kernel = 'rbf', gamma = 15)
X_kpca = kpca.fit_transform(X_scaled)

df_kpca = pd.DataFrame(data = X_kpca, columns=['Component 1', 'Component 2'])
df_kpca['Target'] = y

plt.figure(figsize=(8,6))
for target, color in zip([0,1], ['red','blue']):
    subset = df_kpca[df_kpca['Target'] == target]
    plt.scatter(subset['Component 1'], subset['Component 2'], color = color, label = f"Class {target}")

plt.title("Kernel PCA(RBF) - Non-Linear Dimensionality Reduction")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.show()