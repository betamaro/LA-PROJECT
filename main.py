
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Simulating ratings from 1000 users for 6 genres
np.random.seed(42)
n_users = 1000
n_genres = 6
data = np.random.randint(1, 6, size=(n_users, n_genres))

print("First 5 rows of the dataset:")
print(data[:5])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

print("First 5 rows of the PCA-transformed data:")
print(data_pca[:5])

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', edgecolor='k', s=100)
plt.title('PCA of Movie Genre Ratings (6D to 2D)')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_pca)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=100)
plt.title('Clustering of Users Based on Movie Genre Preferences')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True)
plt.show()

print("Principal Components (loadings):")
print(pca.components_)
