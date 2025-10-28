import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data
y = digits.target

print("Original Data Shape: ", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


tsne = TSNE(n_components=2, perplexity= 30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10,8))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:, 1], c = y, cmap = 'tab10', s = 20)
plt.colorbar(scatter , label = 'Digit Label')
plt.title("t-SNE Visualization of Handwritten Digits (2D Projection)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
