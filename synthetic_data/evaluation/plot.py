import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path

def load_mode_scores_from_summary(
    base_dir: Path,
    emb_type_list: List[dict],
    metrics: Dict[str, callable],
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:

    mode_scores = {
        metric: {
            f"{emb['rep_type']}_reg{emb['regularized']}": {}
            for emb in emb_type_list
        }
        for metric in metrics
    }

    for emb in emb_type_list:
        rep_type = emb["rep_type"]
        regularized = emb["regularized"]
        key = f"{rep_type}_reg{regularized}"
        summary_path = base_dir / f"evaluation_summary_{key}.txt"

        if not summary_path.exists():
            print(f"[Warning] {summary_path} does not exist.")
            continue

        current_metric = None
        with open(summary_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("### Metric:"):
                    current_metric = line.replace("###", "").replace("Metric:", "").strip()
                elif current_metric and line and not line.startswith("#"):
                    try:
                        pq_key, mean_str, _, std_str = line.split()
                        mean = float(mean_str)
                        std = float(std_str)
                        mode_scores[current_metric][key][pq_key] = (mean, std)
                    except ValueError:
                        continue
    return mode_scores


def plot_multi_modes(
    pq_list: List[Tuple[float, float]],
    base_dir: Path,
    emb_type_list: List[dict],
    metrics: Dict[str, callable],
    label_reg: bool = False
):

    mode_scores = load_mode_scores_from_summary(
        base_dir=base_dir,
        emb_type_list=emb_type_list,
        metrics=metrics
    )

    p_values = [p for p, _ in pq_list]
    pq_keys = [f"p{p}_q{q}" for p, q in pq_list]

    markers = ['o', 's', '^', 'D', 'v', '*']
    colors = plt.cm.get_cmap('tab10')

    output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics.keys():
        plt.figure(figsize=(8, 5))
        plt.title(f"{metric} Mean ± Std vs p")
        plt.xlabel("p")
        plt.ylabel(metric)

        for i, emb in enumerate(emb_type_list):
            key = f"{emb['rep_type']}_reg{emb['regularized']}"
            if label_reg:
                label = key
            else:
                label = f"{emb['rep_type']}"
            means, stds = [], []
            for pq in pq_keys:
                mean_std = mode_scores.get(metric, {}).get(key, {}).get(pq, (np.nan, np.nan))
                mean, std = mean_std
                means.append(mean)
                stds.append(std)

            marker = markers[i % len(markers)]
            color = colors(i % 10)
            plt.errorbar(
                p_values, means, yerr=stds,
                label=label, fmt='-' + marker, capsize=5, color=color
            )

        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()

        save_path = output_dir / f"{metric}_vs_p.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {save_path}")
