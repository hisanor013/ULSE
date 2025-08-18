import os
from typing import List, Tuple
from .edge_sampling import generate_edge_probability_matrix, sample_edges, save_edges

def sample_and_save_from_data(
    static_labels,
    dynamic_node_labels,
    p: float,
    q: float,
    output_dir: str,
    run_id: int,
    seed: int,
    connected: bool = False
):
    n = len(static_labels)
    P_list = [generate_edge_probability_matrix(labels, p, q, n) for labels in dynamic_node_labels]
    synthetic_data = sample_edges(P_list, n, seed, connected = connected)
    run_dir = os.path.join(output_dir, f"p{p}_q{q}", f"run{run_id}")
    save_edges(run_dir, synthetic_data)

def batch_sample_and_save(
    static_labels,
    dynamic_node_labels,
    output_dir: str,
    pq_list: List[Tuple[float, float]],
    runs_per_pq: int = 1,
    base_seed: int = 0,
    connected: bool = False
):
    for p, q in pq_list:
        for run_id in range(runs_per_pq):
            seed = base_seed + run_id
            sample_and_save_from_data(
                static_labels=static_labels,
                dynamic_node_labels=dynamic_node_labels,
                p=p,
                q=q,
                output_dir=output_dir,
                run_id=run_id,
                seed=seed,
                connected = connected
            )
