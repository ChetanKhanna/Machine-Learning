import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.60,
                  random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.show()
# using k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_pred = kmeans.predict(X)
# passing extra 'c' arg for coluring clusters differently
plt.scatter(X[:, 0], X[:, 1], s=10, c=y_pred)
plt.show()
