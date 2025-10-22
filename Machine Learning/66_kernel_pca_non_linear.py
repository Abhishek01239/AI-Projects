import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples = 200, noise = 0.1, random_state=42)

plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c = y, cmap  = 'viridis', s = 30)
plt.title("Original Nonlinear Data(Two Moons)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma = 10)
X_kpca =  kpca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_kpca[:, 0], X_kpca[:,1], c = y, cmap = 'plasma', s = 30)
plt.title("Kernel PCA Projection (RBF Keernel)")
plt.xlabel("Principal Components 1")
plt.ylabel("Principal Componenets 2")
plt.show()