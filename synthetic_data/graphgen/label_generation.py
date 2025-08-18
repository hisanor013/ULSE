import numpy as np
from typing import List, Dict, Optional
import os

def assign_static_labels(n: int, K: int, pi: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(K, size=n, p=pi)

def generate_dynamic_labels_from_static(
    static_labels: np.ndarray,
    label_mappings: List[Dict[int, int]]
) -> List[Dict[int, int]]:
    dynamic_node_labels = []
    for mapping in label_mappings:
        time_labels = {
            node: mapping.get(static_labels[node], static_labels[node])
            for node in range(len(static_labels))
        }
        dynamic_node_labels.append(time_labels)
    return dynamic_node_labels

def save_static_node_labels(dir_path: str, static_labels: np.ndarray):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "static_node2label.txt"), "w") as f:
        for node, label in enumerate(static_labels):
            f.write(f"{node} {label}\n")

def save_dynamic_node_labels(dir_path: str, dynamic_node_labels: List[Dict[int, int]]):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "node2label.txt"), "w") as f:
        for t, labels in enumerate(dynamic_node_labels, 1):
            for node, label in labels.items():
                f.write(f"{node} {label}\n")
