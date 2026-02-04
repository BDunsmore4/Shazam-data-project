import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import umap
import hdbscan
import warnings
import sys
import os

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import OUTPUT_DIR


def separate_features_from_metadata(df):
    """Split metadata columns from acoustic features"""

    metadata_patterns = [
        'songID', 'title', 'artist', 'url', 'genre', 'duplicate',
        'variant', 'Unnamed', 'key', 'scale', 'codec', 'md5'
    ]

    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter out metadata
    metadata_cols = []
    for col in df.columns:
        if any(pattern.lower() in col.lower() for pattern in metadata_patterns):
            metadata_cols.append(col)

    acoustic_cols = [col for col in numeric_cols if col not in metadata_cols]

    print(f"Total columns: {len(df.columns)}")
    print(f"Acoustic features: {len(acoustic_cols)}")
    print(f"Metadata: {len(metadata_cols)}\n")

    return acoustic_cols, metadata_cols


def estimate_intrinsic_dim(X, k=20):
    # Estimate data dimensionality using nearest neighbors
    n_samples = min(X.shape[0], 1000)  # subsample for speed
    X_sample = X[:n_samples] if X.shape[0] > 1000 else X

    nbrs = NearestNeighbors(n_neighbors=min(k, X_sample.shape[0] - 1)).fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)

    ratios = distances[:, -1] / (distances[:, 1] + 1e-10)
    dim = int(np.mean(1 / np.log(ratios + 1)))

    return max(2, min(dim, 50))


def optimize_pca(X, variance_threshold=0.95):
    # Find optimal PCA components using a learning model


    n_features = X.shape[1]
    n_samples = X.shape[0]
    max_components = min(100, n_features, n_samples - 1)

    # Fit full PCA
    pca = PCA(n_components=max_components)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Method 1: variance threshold
    n_var = np.argmax(cumvar >= variance_threshold) + 1

    # Method 2: intrinsic dimension
    n_intrinsic = estimate_intrinsic_dim(X)

    # Method 3: elbow (second derivative)
    if len(cumvar) > 5:
        second_deriv = np.diff(cumvar, 2)
        n_elbow = np.argmax(second_deriv > 0) + 2
    else:
        n_elbow = n_var

    # Take median of methods
    n_components = int(np.median([n_var, n_intrinsic, n_elbow]))
    n_components = max(10, min(n_components, max_components))

    print(f"  Variance method: {n_var}")
    print(f"  Intrinsic dim: {n_intrinsic}")
    print(f"  Elbow: {n_elbow}")
    print(f"  Selected: {n_components} (explains {cumvar[n_components - 1]:.1%})\n")

    # Refit with optimal components
    pca_final = PCA(n_components=n_components)
    X_pca = pca_final.fit_transform(X)

    return X_pca, n_components


def evaluate_clustering(X, labels):
    # Calculate clustering quality metrics
    mask = labels >= 0

    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return {'silhouette': -1, 'n_clusters': 0, 'noise_ratio': 1.0}

    X_clean = X[mask]
    labels_clean = labels[mask]

    sil = silhouette_score(X_clean, labels_clean)
    n_clusters = len(np.unique(labels_clean))
    noise_ratio = 1 - np.mean(mask)

    return {
        'silhouette': sil,
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio
    }


def optimize_umap(X_pca, n_components=3, n_trials=12):
    """Grid search for best UMAP parameters"""
    print(f"Optimizing UMAP ({n_trials} configs)...")

    n_samples = X_pca.shape[0]

    # Parameter ranges
    n_neighbors_vals = np.linspace(10, min(50, n_samples // 3), 4, dtype=int)
    min_dist_vals = [0.0, 0.1, 0.3]

    best_score = -np.inf
    best_params = None
    best_embedding = None

    for n_neighbors in n_neighbors_vals:
        for min_dist in min_dist_vals:
            reducer = umap.UMAP(
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                n_components=n_components,
                random_state=42
            )

            embedding = reducer.fit_transform(X_pca)

            # clustering for evaluation
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, n_samples // 50))
            labels = clusterer.fit_predict(embedding)

            metrics = evaluate_clustering(embedding, labels)

            # Simple composite score
            score = metrics['silhouette'] * 2 - metrics['noise_ratio']

            if score > best_score:
                best_score = score
                best_params = {'n_neighbors': int(n_neighbors), 'min_dist': float(min_dist)}
                best_embedding = embedding
                print(f"  Best so far: n_neighbors={n_neighbors}, min_dist={min_dist:.1f}, "
                      f"sil={metrics['silhouette']:.3f}")

    print(f"  Final: {best_params}\n")
    return best_embedding, best_params


def optimize_hdbscan(X):
    """Grid search for best HDBSCAN parameters"""
    print("Optimizing HDBSCAN...")

    n_samples = X.shape[0]

    min_cluster_sizes = np.linspace(5, max(10, n_samples // 30), 5, dtype=int)
    min_samples_vals = [1, 3, 5]

    best_score = -np.inf
    best_params = None
    best_labels = None

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_vals:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples)
            )

            labels = clusterer.fit_predict(X)
            metrics = evaluate_clustering(X, labels)

            # Score clustering
            score = metrics['silhouette'] * 2 - metrics['noise_ratio']

            if score > best_score and metrics['n_clusters'] >= 2:
                best_score = score
                best_params = {
                    'min_cluster_size': int(min_cluster_size),
                    'min_samples': int(min_samples)
                }
                best_labels = labels
                print(f"  Best: mcs={min_cluster_size}, ms={min_samples}, "
                      f"sil={metrics['silhouette']:.3f}, n={metrics['n_clusters']}")

    print(f"  Final: {best_params}\n")
    return best_labels, best_params


def run_clustering_pipeline(X, variance_threshold=0.95, umap_components=3):
    """Complete clustering pipeline with optimization"""

    print(f"\nInput: {X.shape[0]} samples, {X.shape[1]} features\n")

    # Standardize
    print("Standardizing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    X_pca, n_pca = optimize_pca(X_scaled, variance_threshold)

    # UMAP
    X_umap, umap_params = optimize_umap(X_pca, umap_components)

    # HDBSCAN
    labels, hdbscan_params = optimize_hdbscan(X_umap)

    final_metrics = evaluate_clustering(X_umap, labels)

    print(f"Clusters: {final_metrics['n_clusters']}")
    print(f"Silhouette: {final_metrics['silhouette']:.3f}")
    print(f"Noise: {final_metrics['noise_ratio']:.1%}")
    print("=" * 70 + "\n")

    return {
        'labels': labels,
        'umap_embedding': X_umap,
        'pca_components': n_pca,
        'umap_params': umap_params,
        'hdbscan_params': hdbscan_params,
        'metrics': final_metrics
    }


if __name__ == "__main__":
    # Config
    INPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_AND_checked_data.csv")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "post_UMAP.csv")

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded: {df.shape}\n")

    # Separate features from metadata
    acoustic_cols, metadata_cols = separate_features_from_metadata(df)

    X = df[acoustic_cols].fillna(0).values

    if X.shape[0] < 50 or X.shape[1] < 10:
        print("Error: Not enough data for clustering")
        exit(1)

    results = run_clustering_pipeline(X, variance_threshold=0.95, umap_components=3)

    # Add results to dataframe
    df['cluster'] = results['labels']
    for i in range(3):
        df[f'UMAP_{i + 1}'] = results['umap_embedding'][:, i]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")

    # Saving parameters
    params_file = os.path.join(OUTPUT_DIR, "clustering_params.txt")
    with open(params_file, 'w') as f:
        f.write("Best Parameters\n")
        f.write("=" * 50 + "\n")
        f.write(f"PCA components: {results['pca_components']}\n")
        f.write(f"UMAP: {results['umap_params']}\n")
        f.write(f"HDBSCAN: {results['hdbscan_params']}\n")
        f.write(f"\nMetrics: {results['metrics']}\n")

    print(f"Saved: {params_file}")
