import numpy as np
import os
from typing import List, Dict, Tuple, Optional
max_retries = 10

def generate_edge_probability_matrix(
    labels: Dict[int, int],
    p: float,
    q: float,
    n: int
) -> np.ndarray:
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            P[i, j] = p if labels[i] == labels[j] else q
            P[j, i] = P[i, j]
    return P

def sample_edges(
    P_list: List[np.ndarray],
    n_nodes: int,
    seed: Optional[int] = None,
    connected: Optional[bool] = False,
) -> List[Tuple[int, int, int]]:
    for attempt in range(max_retries):
            if seed is not None:
                np.random.seed(seed + attempt)
    
            synthetic_data = []
            for t, P in enumerate(P_list, 1):
                for i in range(P.shape[0]):
                    for j in range(i + 1):
                        if np.random.rand() < P[i, j]:
                            synthetic_data.append((i, j, t))
    
            if connected:
                if not has_isolated_nodes(synthetic_data, n_nodes):
                    return synthetic_data
                else:
                    print(f"🔁 Retry {attempt + 1}: isolated node detected, resampling...")
            else:
                return synthetic_data
    
    print(f"⚠️ Failed to generate connected graph after {max_retries} attempts.")
    return synthetic_data

def save_edges(dir_path: str, synthetic_data: List[Tuple[int, int, int]]):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "synthetic.txt"), "w") as f:
        for src, tgt, t in synthetic_data:
            f.write(f"{src} {tgt} {t}\n")

def has_isolated_nodes(synthetic_data: List[Tuple[int, int, int]], n_nodes: int) -> bool:

    from collections import defaultdict

    time_degrees = defaultdict(lambda: [0] * n_nodes)
    for u, v, t in synthetic_data:
        time_degrees[t][u] += 1
        time_degrees[t][v] += 1

    for t, degrees in time_degrees.items():
        if any(d == 0 for d in degrees):
            return True
    return False

