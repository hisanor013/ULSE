import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from .label_generation import assign_static_labels, generate_dynamic_labels_from_static, save_static_node_labels, save_dynamic_node_labels

def save_config(
    dir_path: str,
    K: int,
    pi: np.ndarray,
    label_mappings: List[Dict[int, int]],
    seed: Optional[int],
    p: Optional[float],
    q: Optional[float]
):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "config.txt"), "w") as f:
        f.write(f"K {K}\n")
        f.write(f"pi {' '.join(map(str, pi))}\n")
        f.write(f"seed {seed}\n")
        if p is not None and q is not None:
            f.write(f"p {p}\n")
            f.write(f"q {q}\n")
        f.write("label_mappings\n")
        for t, mapping in enumerate(label_mappings, 1):
            line = f"time {t}: " + ", ".join(f"{k}->{v}" for k,v in mapping.items())
            f.write(line + "\n")

def generate_and_save_static_config(
    n: int,
    K: int,
    pi: np.ndarray,
    label_mappings: List[Dict[int, int]],
    config_dir: str,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[Dict[int, int]]]:
    static_labels = assign_static_labels(n, K, pi, seed)
    dynamic_node_labels = generate_dynamic_labels_from_static(static_labels, label_mappings)
    save_static_node_labels(config_dir, static_labels)
    save_dynamic_node_labels(config_dir, dynamic_node_labels)
    save_config(config_dir, K, pi, label_mappings, seed, p=None, q=None)
    return static_labels, dynamic_node_labels
