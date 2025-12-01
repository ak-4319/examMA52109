###
## cluster_maker - agglomerative.py
## Athul
## December 2025
###

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
    affinity: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical agglomerative clustering using scikit-learn.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    k : int
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        Linkage criterion.
    affinity : str, default "euclidean"
        Metric used to compute linkage (ignored if linkage="ward").

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    centroids : ndarray of shape (k, n_features)
        Approximate centroids computed as the mean of points in each cluster.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a NumPy array.")
    if k <= 1:
        raise ValueError("Number of clusters k must be greater than 1.")

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        affinity=affinity,
    )
    labels = model.fit_predict(X)

    # Compute approximate centroids
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features), dtype=float)
    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.any(mask):
            centroids[cluster_id] = X[mask].mean(axis=0)
        else:
            centroids[cluster_id] = np.zeros(n_features)

    return labels, centroids