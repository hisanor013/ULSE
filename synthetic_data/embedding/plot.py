import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Tuple
from sklearn.manifold import TSNE

def plot_raw(
    embedding_path: Path,
    label_path: Path,
    offset: float = 0.2,
    columns: List[int] = None,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    save_path: Path = None,
    method_name: str = "",
    n_times: int = None, 
):

    if columns is None:
        columns = [0, 1]
    if columns is None or len(columns) != 2:
        raise ValueError("The 'columns' parameter must specify two dimensions for plotting.")
    col_x, col_y = columns

    with open(embedding_path, "r") as f:
        n_nodes, dim = map(int, f.readline().strip().split())
        lines = f.readlines()

    total_lines = len(lines)
    total_n_times = total_lines // n_nodes
    
    if n_times is None:
        n_times = total_n_times
    elif n_times > total_n_times:
        raise ValueError(f"The specified n_times={n_times} exceeds the total number of time steps available in the file ({total_n_times}).")

    embedding_mat = np.zeros((total_lines, dim), dtype=float)

    for i, line in enumerate(lines):
        parts = line.strip().split()
        embedding_mat[i] = np.array(parts[1:], dtype=float)

    label_lines = Path(label_path).read_text().strip().splitlines()
    labels = np.array([int(line.strip().split()[1]) for line in label_lines])

    if len(labels) != n_nodes:
        raise ValueError(f"The number of labels ({len(labels)}) does not match the number of nodes ({n_nodes}).")

    cluster_ids = np.unique(labels)
    cluster_num = len(cluster_ids)

    CUSTOM_COLORS = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#d62728",  # red
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
    ]
    MARKERS = ["o", "^", "X", "s", "D", "v", "P", "*"]

    colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(cluster_num)]
    markers = [MARKERS[i % len(MARKERS)] for i in range(cluster_num)]

    x_all = embedding_mat[:, col_x]
    y_all = embedding_mat[:, col_y]

    if xlim is None:
        x_margin = (x_all.max() - x_all.min()) * 0.05
        xlim = (x_all.min() - x_margin, x_all.max() + x_margin)
    if ylim is None:
        y_margin = (y_all.max() - y_all.min()) * 0.05
        ylim = (y_all.min() - y_margin, y_all.max() + y_margin)

    fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 5), squeeze=False)

    for t in range(n_times):
        ax = axes[0][t]
        start = t * n_nodes
        end = (t + 1) * n_nodes

        x = embedding_mat[start:end, col_x]
        y = embedding_mat[start:end, col_y]

        for j, cid in enumerate(cluster_ids):
            inds = labels == cid
            ax.scatter(
                x[inds],
                y[inds],
                label=f"Community {cid+1}" if t == 0 else None,
                color=colors[j],
                marker=markers[j],
                s=30,
            )

        ax.set_title(f"{method_name} at Time {t + 1}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if t == 0:
            ax.legend()

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()

def plot_tsne(
    embedding_path: Path,
    label_path: Path,
    perplexity: float = 30,
    random_state: int = 42,
    method_name: str = "",
    columns: List[int] = None,
    save_path: Path = None
):

    with open(embedding_path, "r") as f:
        n_nodes, dim = map(int, f.readline().strip().split())
        lines = f.readlines()

    total_lines = len(lines)
    n_times = total_lines // n_nodes
    embedding_mat = np.zeros((total_lines, dim), dtype=float)

    for i, line in enumerate(lines):
        parts = line.strip().split()
        embedding_mat[i] = np.array(parts[1:], dtype=float)

    label_lines = Path(label_path).read_text().strip().splitlines()
    labels = np.array([int(line.strip().split()[1]) for line in label_lines])

    if columns is not None:
        embedding_mat = embedding_mat[:, columns]

    cluster_ids = np.unique(labels)
    cluster_num = len(cluster_ids)

    CUSTOM_COLORS = [
        "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]
    MARKERS = ["o", "^", "X", "s", "D", "v", "P", "*"]

    colors = [CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(cluster_num)]
    markers = [MARKERS[i % len(MARKERS)] for i in range(cluster_num)]

    fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 5))
    if n_times == 1:
        axes = [axes]

    for t in range(n_times):
        ax = axes[t]
        start = t * n_nodes
        end = (t + 1) * n_nodes
        emb_t = embedding_mat[start:end, :]
        labels_t = labels

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        emb_2d = tsne.fit_transform(emb_t)

        for i, cid in enumerate(cluster_ids):
            inds = labels_t == cid
            ax.scatter(
                emb_2d[inds, 0],
                emb_2d[inds, 1],
                label=f"Community {cid+1}",
                color=colors[i],
                marker=markers[i],
                s=30,
            )

        ax.set_title(f"{method_name} at Time {t + 1}")
        if t == 0:
            ax.legend()

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.show()