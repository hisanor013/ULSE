from typing import List, Dict
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from evaluation.io import load_labels_per_time

def calculate_metrics_for_all_times(
    pred_labels_file: str,
    true_labels_file: str,
    n_nodes: int,
    metrics: dict,
) -> dict:
    """
    metrics = {
      'NMI': sklearn.metrics.normalized_mutual_info_score,
      'ARI': sklearn.metrics.adjusted_rand_score,
    }
    """
    pred_labels_per_time = load_labels_per_time(pred_labels_file, n_nodes)
    true_labels_per_time = load_labels_per_time(true_labels_file, n_nodes)
    
    results = {name: [] for name in metrics.keys()}
    
    for pred, true in zip(pred_labels_per_time, true_labels_per_time):
        pred_list = [pred[node] for node in range(n_nodes)]
        true_list = [true[node] for node in range(n_nodes)]
        for name, func in metrics.items():
            score = func(true_list, pred_list)
            results[name].append(score)
    return results
