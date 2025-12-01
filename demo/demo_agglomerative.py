###
## demo/demo_agglomerative.py
## Demonstration of agglomerative clustering on difficult_dataset.csv
###

import os
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.preprocessing import select_features, standardise_features
from cluster_maker.agglomerative import agglomerative_clustering
from cluster_maker.evaluation import compute_inertia, silhouette_score_sklearn
from cluster_maker.plotting_clustered import plot_clusters_2d

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
DATA_PATH = os.path.join("data", "difficult_dataset.csv")
df = pd.read_csv(DATA_PATH)

# Select numeric features
feature_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
X_df = select_features(df, feature_cols)
X = standardise_features(X_df.to_numpy(dtype=float))

# ---------------------------------------------------------------------
# Run agglomerative clustering
# ---------------------------------------------------------------------
k = 4  # plausible number of clusters for difficult dataset
labels, centroids = agglomerative_clustering(X, k=k, linkage="ward")

# Compute metrics
inertia = compute_inertia(X, labels, centroids)
try:
    silhouette = silhouette_score_sklearn(X, labels)
except ValueError:
    silhouette = None

print("=== Agglomerative Clustering Analysis ===")
print(f"Chosen k: {k}")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette score: {silhouette:.3f}" if silhouette is not None else "Silhouette score: N/A")

# ---------------------------------------------------------------------
# Add labels to DataFrame and export
# ---------------------------------------------------------------------
df = df.copy()
df["cluster"] = labels
os.makedirs("demo_output", exist_ok=True)
df.to_csv(os.path.join("demo_output", "agglomerative_clusters.csv"), index=False)

# ---------------------------------------------------------------------
# Plot clusters
# ---------------------------------------------------------------------
fig_cluster, ax_cluster = plot_clusters_2d(X, labels, centroids=centroids,
                                           title=f"Agglomerative Clustering (k={k})")
fig_cluster.savefig(os.path.join("demo_output", f"agglomerative_clusters_k{k}.png"))

# ---------------------------------------------------------------------
# Supporting visualisation: dendrogram (optional)
# ---------------------------------------------------------------------
from scipy.cluster.hierarchy import linkage, dendrogram

fig_dendro, ax_dendro = plt.subplots(figsize=(8, 5))
Z = linkage(X, method="ward")
dendrogram(Z, truncate_mode="level", p=5, ax=ax_dendro)
ax_dendro.set_title("Agglomerative Clustering Dendrogram")
ax_dendro.set_xlabel("Sample index")
ax_dendro.set_ylabel("Distance")
fig_dendro.tight_layout()
fig_dendro.savefig(os.path.join("demo_output", "agglomerative_dendrogram.png"))