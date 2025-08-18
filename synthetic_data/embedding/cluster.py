import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from typing import List

def cluster_embeddings_kmeans_and_save(
    embedding_path: Path,
    clusters_per_time: List[int],
    output_path: Path,
    random_state: int = 0,
):
    with open(embedding_path, "r") as f:
        n_nodes, dim = map(int, f.readline().strip().split())
        lines = f.readlines()

    total_lines = len(lines)
    n_times = total_lines // n_nodes
    embeddings = np.zeros((total_lines, dim), dtype=float)

    for i, line in enumerate(lines):
        parts = line.strip().split()
        embeddings[i] = np.array(parts[1:], dtype=float)

    labels_all_times = []

    for t in range(n_times):
        start_idx = t * n_nodes
        end_idx = (t + 1) * n_nodes
        X = embeddings[start_idx:end_idx]
        k = clusters_per_time[t]
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        labels_all_times.append(kmeans.labels_)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t, labels in enumerate(labels_all_times, 1):
            for node, label in enumerate(labels):
                f.write(f"{node} {label}\n")
