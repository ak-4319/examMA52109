import os
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.preprocessing import select_features, standardise_features
from cluster_maker.algorithms import kmeans
from cluster_maker.evaluation import compute_inertia, silhouette_score_sklearn, elbow_curve
from cluster_maker.plotting_clustered import plot_clusters_2d, plot_elbow
from cluster_maker.data_exporter import export_to_csv
# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
DATA_PATH = os.path.join("data", "simulated_data.csv")
df = pd.read_csv(DATA_PATH)

# Select all numeric columns as features
feature_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
X_df = select_features(df, feature_cols)
X = X_df.to_numpy(dtype=float)   # convert DataFrame → NumPy array
X = standardise_features(X)
# ---------------------------------------------------------------------
# Step 1: Explore plausible cluster numbers using elbow method
# ---------------------------------------------------------------------
k_values = list(range(2, 8))  # try 2–7 clusters
inertias = elbow_curve(X, k_values)

fig_elbow, ax_elbow = plot_elbow(k_values, [inertias[k] for k in k_values])
fig_elbow.suptitle("Elbow Curve for Simulated Data")
fig_elbow.savefig(os.path.join("demo_output", "elbow_curve.png"))

# ---------------------------------------------------------------------
# Step 2: Choose a plausible k based on elbow + silhouette
# ---------------------------------------------------------------------
best_k = None
best_score = -1
scores = {}

for k in k_values:
    labels, centroids = kmeans(X, k=k, random_state=42)
    try:
        sil = silhouette_score_sklearn(X, labels)
    except ValueError:
        sil = None
    scores[k] = sil
    if sil is not None and sil > best_score:
        best_score = sil
        best_k = k

# ---------------------------------------------------------------------
# Step 3: Final clustering with chosen k
# ---------------------------------------------------------------------
labels, centroids = kmeans(X, k=best_k, random_state=42)
inertia = compute_inertia(X, labels, centroids)

# Add cluster labels to DataFrame
df = df.copy()
df["cluster"] = labels

# Export clustered dataset to CSV
clustered_csv_path = os.path.join("demo_output", f"clustered_simulated_data_k{best_k}.csv")
export_to_csv(df, clustered_csv_path, delimiter=",", include_index=False)

# Plot clusters in 2D
fig_cluster, ax_cluster = plot_clusters_2d(X, labels, centroids=centroids,
                                           title=f"Cluster plot (k={best_k})")
fig_cluster.savefig(os.path.join("demo_output", f"clusters_k{best_k}.png"))

# ---------------------------------------------------------------------
# Step 4: Supporting visualisation – silhouette scores vs k
# ---------------------------------------------------------------------
fig_sil, ax_sil = plt.subplots()
ax_sil.bar(scores.keys(), [s if s is not None else 0 for s in scores.values()])
ax_sil.set_xlabel("Number of clusters (k)")
ax_sil.set_ylabel("Silhouette score")
ax_sil.set_title("Silhouette scores for different k")
fig_sil.tight_layout()
fig_sil.savefig(os.path.join("demo_output", "silhouette_scores.png"))

# ---------------------------------------------------------------------
# Step 5: Print summary
# ---------------------------------------------------------------------
print("=== Simulated Clustering Analysis ===")
print(f"Chosen k: {best_k}")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette score: {best_score:.3f}")
print("Clustered data saved with labels in demo_output/")