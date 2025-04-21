import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Page title
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Sidebar - Number of clusters
st.sidebar.header("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 4)  # Default set to 4

# Run K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

# Reduce dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(kmeans.cluster_centers_)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='black', marker='X', label='Centroids')
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(*scatter.legend_elements(), title="Cluster")
st.pyplot(fig)

# Show raw data (with cluster info)
clustered_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
clustered_df["Cluster"] = labels
st.dataframe(clustered_df.head(10))
