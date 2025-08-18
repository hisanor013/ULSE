from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from evaluation.metrics import calculate_metrics_for_all_times
from evaluation.io import load_labels_per_time
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI

def run_evaluation_pipeline(
    base_dir: Path,
    pq_list: List[Tuple[float, float]],
    runs_per_pq: int,
    n_nodes: int,
    true_label_path: Path,
    emb_type_list: List[dict] = [{"rep_type": "UASE", "regularized": False}, 
                                {"rep_type": "ULSE-n1", "regularized": True},
                                {"rep_type": "ULSE-n2", "regularized": True}],
    metrics: dict = {"NMI": NMI, "ARI": ARI},
):

    prediction_file_names = [
        f"predicted_labels_{emb['rep_type']}_reg{emb['regularized']}.txt"
        for emb in emb_type_list
    ]

    results = {
        metric: {fname: {} for fname in prediction_file_names}
        for metric in metrics
    }

    for p, q in pq_list:
        pq_dir = base_dir / f"p{p}_q{q}"
        evaluation_per_pq = {
            pred_file: {metric: [] for metric in metrics} for pred_file in prediction_file_names
        }

        for emb in emb_type_list:
            rep_type = emb["rep_type"]
            regularized = emb["regularized"]
            pred_file_name = f"predicted_labels_{rep_type}_reg{regularized}.txt"
            for run_id in range(runs_per_pq):
                run_dir = pq_dir / f"run{run_id}"
                pred_label_file = run_dir / pred_file_name
                true_label_file = true_label_path
                if not pred_label_file.exists() or not true_label_file.exists():
                    continue

                metric_scores = calculate_metrics_for_all_times(
                    pred_labels_file=str(pred_label_file),
                    true_labels_file=str(true_label_file),
                    n_nodes=n_nodes,
                    metrics=metrics,
                )
                for metric_name, scores in metric_scores.items():
                    avg_score = np.mean(scores)
                    results[metric_name][pred_file_name].setdefault(f"p{p}_q{q}", []).append(avg_score)

                    evaluation_per_pq[pred_file_name][metric_name].append({
                        "run": run_id,
                        "scores_per_time": scores,
                        "average_score": avg_score
                    })

            eval_file_name = f"evaluation_{rep_type}_reg{regularized}.txt"
            eval_path = pq_dir / eval_file_name
            with open(eval_path, "w") as f:
                f.write(f"## Evaluation for rep_type: {rep_type}, regularized: {regularized}, p: {p}, q: {q} ##\n\n")
                for metric_name in metrics:
                    f.write(f"### Metric: {metric_name} ###\n")
                    runs_data = evaluation_per_pq[pred_file_name][metric_name]
                    if not runs_data:
                        f.write("No data available.\n")
                        continue
                    for run_data in runs_data:
                        run_id = run_data["run"]
                        scores_per_time = run_data["scores_per_time"]
                        avg_score = run_data["average_score"]
                        scores_str = ", ".join(f"{score:.4f}" for score in scores_per_time)
                        f.write(f"Run {run_id}: Time scores: [{scores_str}] | Average: {avg_score:.12f}\n")
                    f.write("\n")

    for emb in emb_type_list:
        rep_type = emb["rep_type"]
        regularized = emb["regularized"]
        pred_file_name = f"predicted_labels_{rep_type}_reg{regularized}.txt"
        summary_path = base_dir / f"evaluation_summary_{rep_type}_reg{regularized}.txt"

        with open(summary_path, "w") as f:
            f.write(f"## Summary for {rep_type}, regularized={regularized} ##\n\n")
            for metric_name in metrics:
                f.write(f"### Metric: {metric_name} ###\n")
                pq_scores = results[metric_name].get(pred_file_name, {})
                if not pq_scores:
                    f.write("No data available.\n\n")
                    continue
                for pq_key, scores in pq_scores.items():
                    mean = np.mean(scores)
                    std = np.std(scores)
                    f.write(f"{pq_key} {mean:.12f} ± {std:.12f}\n")
                f.write("\n")

    return results
