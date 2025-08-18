from pathlib import Path
from typing import List, Dict, Tuple, Callable
from scipy.stats import ttest_ind_from_stats
from evaluation.plot import load_mode_scores_from_summary

def compute_p_values(
    base_dir: Path,
    emb_type_list: List[dict],
    metrics: Dict[str, Callable],
    sample_size: int,
    emb_high_key: str,
    emb_low_key: str,
) -> None:

    mode_scores = load_mode_scores_from_summary(base_dir, emb_type_list, metrics)
    output_lines = []

    for metric, emb_dict in mode_scores.items():
        if emb_high_key not in emb_dict or emb_low_key not in emb_dict:
            continue

        pq_keys = emb_dict[emb_high_key].keys()

        for pq_key in pq_keys:
            if pq_key not in emb_dict[emb_low_key]:
                continue

            mean_high, std_high = emb_dict[emb_high_key][pq_key]
            mean_low, std_low = emb_dict[emb_low_key][pq_key]

            _, p_val_two_tailed = ttest_ind_from_stats(
                mean1=mean_high, std1=std_high, nobs1=sample_size,
                mean2=mean_low, std2=std_low, nobs2=sample_size,
                equal_var=False
            )

            p_val_one_tailed = p_val_two_tailed / 2

            direction = ">" if mean_high > mean_low else "<"

            line = f"{metric}\t{pq_key}\tp={p_val_one_tailed:.12f}\t({emb_high_key} {direction} {emb_low_key})"
            output_lines.append(line)

    filename = f"one_tailed_p_values_{emb_high_key}_vs_{emb_low_key}.txt"
    output_path = base_dir / filename
    with open(output_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"[INFO] One-sided p-values have been written to {output_path}.")

