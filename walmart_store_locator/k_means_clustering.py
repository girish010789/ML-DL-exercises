import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

##load dataset and plot it as a scatterplot
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
pl = plt.scatter(X[:,0], X[:,1], s=50)
plt.show(pl)

##k-means algorithm, assign 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

##import library
def find_clusters(X, n_clusters, rseed=2):
	#step1: randomly choose clusters
	rng = np.random.RandomState(rseed)
	i = rng.permutation(X.shape[0])[:n_clusters]
	centers = X[i]
	while True:
		##step2: assign labels based on closest center
		labels = pairwise_distances_argmin(X,centers)

		##step3: find new centers from means of points
		new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
	##step4 check for convergence
		if np.all(centers == new_centers):
			break
		centers = new_centers
	return centers,labels

pl1 = []
centers, labels = find_clusters(X,4)
pl1 = plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap='viridis')
pl2 = plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
plt.show(pl1)
plt.show(pl2)
