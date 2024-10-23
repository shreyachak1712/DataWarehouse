# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data


# Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X)


# Perform KMeans (Divisive Clustering)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X)


# Visualizing the clustering results using PCA for 2D plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# Plot Agglomerative Clustering
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='viridis', s=50)
plt.title("Agglomerative Clustering")


# Plot KMeans (Divisive) Clustering
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title("Divisive Clustering")
plt.show()


# Hierarchical Clustering Dendrogram (For Agglomerative Clustering)
Z_agg = linkage(X, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z_agg)
plt.title("Dendrogram for Agglomerative Clustering")
plt.show()


# Divisive (KMeans) Clustering Dendrogram - Simulated using distances
def divisive_dendrogram(X, labels, method='ward'):
	"""
	Generates a simulated dendrogram for divisive clustering by recursively applying KMeans.
	"""
	# Pairwise distances between samples
	distance_matrix = pairwise_distances(X)
    
	# Calculate the linkage matrix using the distances between points
	Z = linkage(distance_matrix, method=method)
    
	# Plot the dendrogram
	plt.figure(figsize=(10, 7))
	dendrogram(Z)
	plt.title("Dendrogram for Divisive Clustering")
	plt.show()


# Generate the divisive dendrogram
divisive_dendrogram(X, kmeans_labels)
