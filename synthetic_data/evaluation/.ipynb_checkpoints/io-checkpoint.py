from typing import Dict, List

def load_labels(label_file: str) -> Dict[int, int]:
    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            node, label = line.strip().split()
            labels[int(node)] = int(label)
    return labels

def load_labels_per_time(label_file: str, n_nodes: int) -> List[Dict[int, int]]:
    with open(label_file, "r") as f:
        lines = f.readlines()
    n_times = len(lines) // n_nodes
    labels_per_time = []
    for t in range(n_times):
        time_labels = {}
        for i in range(n_nodes):
            node, label = lines[t * n_nodes + i].strip().split()
            time_labels[int(node)] = int(label)
        labels_per_time.append(time_labels)
    return labels_per_time

def summarize_results_to_file(nmi_dict: dict, output_path: str):
    with open(output_path, "w") as f:
        for key, scores in nmi_dict.items():
            avg_score = np.mean(scores)
            f.write(f"{key} {avg_score:.4f}\n")