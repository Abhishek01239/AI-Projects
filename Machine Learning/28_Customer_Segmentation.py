import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [15, 39], [16,81], [17, 6], [18, 77], [19, 40], [20, 76],[21, 6], [22, 94] 
])

kmeans = KMeans(n_clusters = 2, random_state = 42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c = labels, cmap = 'viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = "red", marker ='X')
plt.title("Customer Segmentation")
plt.show()
