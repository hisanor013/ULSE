# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import csv
import pandas as pd
from sklearn.cluster import KMeans
from evaluation import evaluation
import sys
import importlib.util

# Add synthetic_data/evaluation directory to path
evaluation_path = os.path.join(
    os.path.dirname(__file__), "../../synthetic_data/evaluation"
)
sys.path.insert(0, evaluation_path)

# Import custom io module
spec = importlib.util.spec_from_file_location(
    "custom_io", os.path.join(evaluation_path, "io.py")
)
custom_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_io)

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI


def survey_available_nodes_per_method(embedding_paths, n_nodes, n_times):
    """
    Survey available (node, time) pairs for each method

    Args:
        embedding_paths: Dictionary of {method_name: file_path}
        n_nodes: Number of nodes
        n_times: Number of time steps

    Returns:
        available_nodes_per_method: {method: {time: set of node_ids}}
    """
    available_nodes_per_method = {}

    for method_name, emb_path in embedding_paths.items():
        if not os.path.exists(emb_path):
            continue

        available_nodes_per_method[method_name] = {}

        try:
            with open(emb_path, "r") as reader:
                reader.readline()  # Skip header line

                line_idx = 0
                for line in reader:
                    embeds = np.fromstring(line.strip(), dtype=float, sep=" ")
                    node_id = int(embeds[0])

                    # Calculate time from line index (0-indexed)
                    time_idx = line_idx // n_nodes

                    # Record available nodes by time
                    if time_idx not in available_nodes_per_method[method_name]:
                        available_nodes_per_method[method_name][time_idx] = set()
                    available_nodes_per_method[method_name][time_idx].add(node_id)

                    line_idx += 1

        except Exception as e:
            print(f"Warning: Error during survey of {method_name}: {e}")
            available_nodes_per_method[method_name] = {}

    return available_nodes_per_method


def load_embeddings_with_time(emb_path, n_nodes, n_times):
    """
    Load embedding file considering time axis

    Args:
        emb_path: Path to embedding file
        n_nodes: Number of nodes
        n_times: Number of time steps

    Returns:
        embeddings_per_time: {time: {node_id: embedding}} dictionary
    """
    embeddings_per_time = {}

    with open(emb_path, "r") as reader:
        reader.readline()  # Skip header line

        line_idx = 0
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=" ")
            node_id = int(embeds[0])
            embedding = embeds[1:]

            # Calculate time from line index (0-indexed)
            time_idx = line_idx // n_nodes

            # Store in time-based dictionary
            if time_idx not in embeddings_per_time:
                embeddings_per_time[time_idx] = {}
            embeddings_per_time[time_idx][node_id] = embedding

            line_idx += 1

    return embeddings_per_time


def test_external_score(
    emb_path, label_path, k, method_name, n_nodes=None, available_nodes=None
):
    """
    Execute clustering evaluation on a single embedding file and return scores
    Args:
        emb_path: Path to embedding file
        label_path: Path to label file
        k: Maximum number of clusters (adjusted according to actual label count)
        method_name: Method name (for display)
        n_nodes: Number of nodes (required for time-wise evaluation)
        available_nodes: Available nodes for this method {time: set of node_ids}
    Returns:
        Dictionary of scores { 'acc': score, 'nmi': score, ... }
    """
    if not os.path.exists(emb_path):
        print(f"Warning: {emb_path} does not exist. Skipping.")
        return None

    if not os.path.exists(label_path):
        print(f"Warning: {label_path} does not exist. Skipping.")
        return None

    # Load labels per time
    try:
        true_labels_per_time = custom_io.load_labels_per_time(label_path, n_nodes)
    except Exception as e:
        print(f"Warning: Failed to load labels per time: {e}")
        # Fallback to conventional method
        return test_external_score_legacy(emb_path, label_path, k, method_name)

    # Get number of time steps
    n_times = len(true_labels_per_time)

    # Load embedding file considering time axis
    try:
        embeddings_per_time = load_embeddings_with_time(emb_path, n_nodes, n_times)
    except Exception as e:
        print(f"Warning: Failed to load embeddings: {e}")
        return None

    if len(embeddings_per_time) == 0:
        print(f"Warning: {method_name} - No valid embeddings found.")
        return None

    # Time-wise clustering evaluation (using only available nodes)
    all_nmi_scores = []
    all_ari_scores = []
    total_evaluated_nodes = 0

    for t, true_labels in enumerate(true_labels_per_time):
        # Get embeddings for this time step
        if t not in embeddings_per_time:
            continue

        time_embeddings = embeddings_per_time[t]

        # Apply available node constraints
        if available_nodes and t in available_nodes:
            available_node_set = available_nodes[t]
        else:
            available_node_set = set(time_embeddings.keys())

        # Extract only evaluation target nodes for this time step
        X = []
        Y = []
        valid_nodes = []

        for node_id in available_node_set:
            if (
                node_id in time_embeddings
                and node_id in true_labels
                and node_id < n_nodes
            ):  # Only nodes within n_nodes range
                X.append(time_embeddings[node_id])
                Y.append(true_labels[node_id])
                valid_nodes.append(node_id)

        if len(X) == 0:
            continue

        total_evaluated_nodes += len(X)

        # Check actual number of labels and adjust cluster count
        unique_labels = set(Y)
        actual_k = min(k, len(unique_labels))

        if len(unique_labels) < 2:
            # Skip if there's only one cluster
            continue

        # Execute K-means++
        model = KMeans(
            n_clusters=actual_k, n_init=100, init="k-means++", random_state=None
        )
        cluster_id = model.fit_predict(X)

        # Calculate evaluation metrics
        nmi_score = NMI(Y, cluster_id)
        ari_score = ARI(Y, cluster_id)

        all_nmi_scores.append(nmi_score)
        all_ari_scores.append(ari_score)

    if len(all_nmi_scores) == 0:
        print(f"Warning: {method_name} - No valid evaluation results.")
        return None

    # Calculate average scores
    avg_nmi = np.mean(all_nmi_scores)
    avg_ari = np.mean(all_ari_scores)

    # Calculate ACC and F1 as average over time
    acc, _, _, f1 = calculate_accuracy_f1_over_time(
        embeddings_per_time, true_labels_per_time, k, n_nodes, available_nodes
    )

    # Calculate average of actually used cluster counts
    actual_clusters_used = []
    for t, true_labels in enumerate(true_labels_per_time):
        if t not in embeddings_per_time:
            continue

        time_embeddings = embeddings_per_time[t]

        # Apply available node constraints
        if available_nodes and t in available_nodes:
            available_node_set = available_nodes[t]
        else:
            available_node_set = set(time_embeddings.keys())

        Y = []
        for node_id in available_node_set:
            if (
                node_id in time_embeddings
                and node_id in true_labels
                and node_id < n_nodes
            ):
                Y.append(true_labels[node_id])
        if len(Y) >= 2:  # At least 2 samples required
            actual_clusters_used.append(len(set(Y)))

    avg_actual_clusters = np.mean(actual_clusters_used) if actual_clusters_used else k

    results = {
        "method": method_name,
        "acc": acc,
        "nmi": avg_nmi,
        "ari": avg_ari,
        "f1": f1,
        "num_nodes": total_evaluated_nodes,
        "num_clusters": avg_actual_clusters,
        "num_time_steps": len(all_nmi_scores),
    }

    # Display results
    print(
        f"{method_name:30s} | ACC: {acc:.4f} | "
        f"NMI: {avg_nmi:.4f} | "
        f"ARI: {avg_ari:.4f} | "
        f"F1: {f1:.4f} | Nodes: {total_evaluated_nodes} | "
        f"Clusters: {avg_actual_clusters:.1f} | "
        f"TimeSteps: {len(all_nmi_scores)}"
    )

    return results


def calculate_accuracy_f1_over_time(
    embeddings_per_time, true_labels_per_time, k, n_nodes, available_nodes=None
):
    """
    Calculate ACC and F1 scores over time (using only available nodes)
    """
    all_acc_scores = []
    all_f1_scores = []

    for t, true_labels in enumerate(true_labels_per_time):
        if t not in embeddings_per_time:
            continue

        time_embeddings = embeddings_per_time[t]

        # Apply available node constraints
        if available_nodes and t in available_nodes:
            available_node_set = available_nodes[t]
        else:
            available_node_set = set(time_embeddings.keys())

        X = []
        Y = []

        for node_id in available_node_set:
            if (
                node_id in time_embeddings
                and node_id in true_labels
                and node_id < n_nodes
            ):
                X.append(time_embeddings[node_id])
                Y.append(true_labels[node_id])

        if len(X) == 0:
            continue

        # Check actual number of labels and adjust cluster count
        unique_labels = set(Y)
        actual_k = min(k, len(unique_labels))

        if len(unique_labels) < 2:
            continue

        # Execute K-means++
        model = KMeans(
            n_clusters=actual_k, n_init=100, init="k-means++", random_state=None
        )
        cluster_id = model.fit_predict(X)

        # Calculate ACC and F1
        acc, _, _, f1 = evaluation(Y, cluster_id)
        all_acc_scores.append(acc)
        all_f1_scores.append(f1)

    if len(all_acc_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0

    return np.mean(all_acc_scores), 0.0, 0.0, np.mean(all_f1_scores)


def test_external_score_legacy(emb_path, label_path, k, method_name):
    """
    Clustering evaluation using conventional method (for fallback)
    """
    n2l_temporal = dict()  # {node_id: [label1, label2, ...]}
    with open(label_path, "r") as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            if n_id not in n2l_temporal:
                n2l_temporal[n_id] = []
            n2l_temporal[n_id].append(l_id)

    node_emb = dict()
    with open(emb_path, "r") as reader:
        reader.readline()  # Skip header line
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=" ")
            node_id = int(embeds[0])
            if node_id in n2l_temporal:
                node_emb[node_id] = embeds[1:]

    Y = []
    X = []
    for node_id in sorted(node_emb.keys()):
        if node_id in n2l_temporal:
            embeddings = node_emb[node_id]
            labels = n2l_temporal[node_id]
            for label in labels:
                Y.append(label)
                X.append(embeddings)

    if len(X) == 0:
        print(f"Warning: {method_name} - No valid nodes found.")
        return None

    # Check actual number of labels and adjust cluster count
    unique_labels = set(Y)
    actual_k = min(k, len(unique_labels))

    if actual_k != k:
        print(f"Warning: Adjusted cluster count from {k} to {actual_k}.")

    # Execute K-means++ (optimized with n_init=100)
    model = KMeans(n_clusters=actual_k, n_init=100, init="k-means++", random_state=None)
    cluster_id = model.fit_predict(X)
    acc, nmi, ari, f1 = evaluation(Y, cluster_id)

    results = {
        "method": method_name,
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "f1": f1,
        "num_nodes": len(X),
        "num_clusters": actual_k,
        "num_time_steps": 1,  # Conventional method is single time step
    }

    # Display results
    print(
        f"{method_name:30s} | ACC: {acc:.4f} | "
        f"NMI: {nmi:.4f} | "
        f"ARI: {ari:.4f} | "
        f"F1: {f1:.4f} | Nodes: {len(X)} | "
        f"Clusters: {actual_k} | "
        f"TimeSteps: 1 (legacy)"
    )

    return results


def find_embedding_files(dataset):
    """
    Detect all .emb files from the emb folder of the specified dataset

    Args:
        dataset: Dataset name

    Returns:
        Dictionary of {method_name: file_path}
    """
    import glob
    import os

    embedding_paths = {}
    emb_dir = f"../../emb/{dataset}"

    if not os.path.exists(emb_dir):
        print(f"Warning: {emb_dir} does not exist.")
        return embedding_paths

    # Search for .emb files
    pattern = os.path.join(emb_dir, "*.emb")
    emb_files = glob.glob(pattern)

    for emb_file in emb_files:
        # Extract method name from filename
        filename = os.path.basename(emb_file)
        # Remove extension
        method_name = filename.replace(".emb", "")
        # Remove dataset name prefix
        if method_name.startswith(f"{dataset}_"):
            method_name = method_name[len(f"{dataset}_") :]

        embedding_paths[method_name] = emb_file

    return embedding_paths


def sort_methods(methods):
    """
    Sort method names in specified order
    - ULSE-n1, ULSE-n2, UASE first
    - Others in alphabetical order
    """
    priority_methods = ["ULSE-n1", "ULSE-n2", "UASE"]

    # High priority methods first
    sorted_methods = []
    for priority_method in priority_methods:
        if priority_method in methods:
            sorted_methods.append(priority_method)

    # Add remaining methods in alphabetical order
    remaining_methods = [m for m in methods if m not in priority_methods]
    sorted_methods.extend(sorted(remaining_methods))

    return sorted_methods


def save_results_to_csv(all_results, output_path):
    """
    Save results to CSV file

    Args:
        all_results: List of all results
        output_path: Save destination path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "dataset",
            "method",
            "acc",
            "nmi",
            "ari",
            "f1",
            "num_nodes",
            "num_clusters",
            "num_time_steps",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"Results saved to {output_path}.")


def save_summary_to_csv(all_scores, output_path):
    """
    Save average performance by method to CSV file

    Args:
        all_scores: Dictionary of scores by method
        output_path: Save destination path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "method",
            "avg_acc",
            "avg_nmi",
            "avg_ari",
            "avg_f1",
            "num_datasets",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for method, scores in all_scores.items():
            avg_acc = np.mean(scores["acc"])
            avg_nmi = np.mean(scores["nmi"])
            avg_ari = np.mean(scores["ari"])
            avg_f1 = np.mean(scores["f1"])
            count = len(scores["acc"])

            writer.writerow(
                {
                    "method": method,
                    "avg_acc": avg_acc,
                    "avg_nmi": avg_nmi,
                    "avg_ari": avg_ari,
                    "avg_f1": avg_f1,
                    "num_datasets": count,
                }
            )

    print(f"Summary saved to {output_path}.")


def format_result_table(all_results, all_scores):
    """
    Format and output results in table format

    Args:
        all_results: List of all results
        all_scores: Dictionary of scores by method
    """
    # Get list of datasets and methods
    datasets = sorted(set(result["dataset"] for result in all_results))
    methods = sort_methods(list(all_scores.keys()))
    metrics = ["acc", "nmi", "ari", "f1"]

    print("\n" + "=" * 120)
    print("Table: Node clustering results in datasets")
    print("=" * 120)

    # Header row
    header = f"{'Data':<12} | {'Metric':<8} |"
    for method in methods:
        header += f" {method:<12} |"
    print(header)
    print("-" * (len(header) + 10))

    # Display results for each dataset-metric combination
    for dataset in datasets:
        for metric in metrics:
            row = f"{dataset:<12} | {metric.upper():<8} |"

            # Collect results for this dataset-metric combination
            metric_results = {}
            for method in methods:
                # Get results for this dataset
                dataset_results = [
                    r
                    for r in all_results
                    if r["dataset"] == dataset and r["method"] == method
                ]
                if dataset_results:
                    metric_results[method] = dataset_results[0][metric]
                else:
                    metric_results[method] = None

            # Create ranking with only valid results
            valid_results = {k: v for k, v in metric_results.items() if v is not None}
            if valid_results:
                # Sort by score (descending)
                sorted_methods = sorted(
                    valid_results.keys(), key=lambda x: valid_results[x], reverse=True
                )
                best_method = sorted_methods[0]
                second_best_method = (
                    sorted_methods[1] if len(sorted_methods) > 1 else None
                )

                for method in methods:
                    if method in metric_results:
                        value = metric_results[method]
                        if value is not None:
                            formatted_value = f"{value:.4f}"
                            if method == best_method:
                                formatted_value = f"**{formatted_value}**"  # Bold
                            elif method == second_best_method:
                                formatted_value = f"__{formatted_value}__"  # Underline
                            row += f" {formatted_value:<12} |"
                        else:
                            row += f" {'N/A':<12} |"
                    else:
                        row += f" {'N/A':<12} |"
            else:
                # When there are no valid results
                for method in methods:
                    row += f" {'N/A':<12} |"

            print(row)

    print("-" * (len(header) + 10))
    print(
        "Note: **bold** = best result, __underline__ = second best result, N/A = Not Available"
    )


def save_table_to_csv(all_results, all_scores, output_path):
    """
    Save table format results to CSV file

    Args:
        all_results: List of all results
        all_scores: Dictionary of scores by method
        output_path: Save destination path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    datasets = sorted(set(result["dataset"] for result in all_results))
    methods = sort_methods(list(all_scores.keys()))
    metrics = ["acc", "nmi", "ari", "f1"]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["dataset", "metric"] + methods
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for dataset in datasets:
            for metric in metrics:
                row = {"dataset": dataset, "metric": metric.upper()}

                # Collect results for this dataset-metric combination
                metric_results = {}
                for method in methods:
                    dataset_results = [
                        r
                        for r in all_results
                        if r["dataset"] == dataset and r["method"] == method
                    ]
                    if dataset_results:
                        metric_results[method] = dataset_results[0][metric]
                    else:
                        metric_results[method] = None

                # Create ranking with only valid results
                valid_results = {
                    k: v for k, v in metric_results.items() if v is not None
                }
                if valid_results:
                    sorted_methods = sorted(
                        valid_results.keys(),
                        key=lambda x: valid_results[x],
                        reverse=True,
                    )
                    best_method = sorted_methods[0]
                    second_best_method = (
                        sorted_methods[1] if len(sorted_methods) > 1 else None
                    )

                    for method in methods:
                        if method in metric_results:
                            value = metric_results[method]
                            if value is not None:
                                formatted_value = f"{value:.4f}"
                                if method == best_method:
                                    formatted_value = f"**{formatted_value}**"
                                elif method == second_best_method:
                                    formatted_value = f"__{formatted_value}__"
                                row[method] = formatted_value
                            else:
                                row[method] = "N/A"
                        else:
                            row[method] = "N/A"
                else:
                    for method in methods:
                        row[method] = "N/A"

                writer.writerow(row)

    print(f"Table format results saved to {output_path}.")


def save_table_to_latex(all_results, all_scores, output_path):
    """
    Save table format results to LaTeX file

    Args:
        all_results: List of all results
        all_scores: Dictionary of scores by method
        output_path: Save destination path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    datasets = sorted(set(result["dataset"] for result in all_results))
    methods = sort_methods(list(all_scores.keys()))
    metrics = ["acc", "nmi", "ari", "f1"]

    # Prepare data for DataFrame
    table_data = []

    for dataset in datasets:
        for metric in metrics:
            row = {"Data": dataset, "Metric": metric.upper()}

            # Collect results for this dataset-metric combination
            metric_results = {}
            for method in methods:
                dataset_results = [
                    r
                    for r in all_results
                    if r["dataset"] == dataset and r["method"] == method
                ]
                if dataset_results:
                    metric_results[method] = dataset_results[0][metric]
                else:
                    metric_results[method] = None

            # Create ranking with only valid results
            valid_results = {k: v for k, v in metric_results.items() if v is not None}
            if valid_results:
                sorted_methods = sorted(
                    valid_results.keys(),
                    key=lambda x: valid_results[x],
                    reverse=True,
                )
                best_method = sorted_methods[0]
                second_best_method = (
                    sorted_methods[1] if len(sorted_methods) > 1 else None
                )

                for method in methods:
                    if method in metric_results:
                        value = metric_results[method]
                        if value is not None:
                            formatted_value = f"{value:.4f}"
                            if method == best_method:
                                formatted_value = f"\\textbf{{{formatted_value}}}"
                            elif method == second_best_method:
                                formatted_value = f"\\underline{{{formatted_value}}}"
                            row[method] = formatted_value
                        else:
                            row[method] = "N/A"
                    else:
                        row[method] = "N/A"
            else:
                for method in methods:
                    row[method] = "N/A"

            table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Save to LaTeX file
    latex_content = df.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
        caption="Node clustering results in datasets. We bold the best results and underline the second best results.",
        label="tab:clustering_results",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"LaTeX format results saved to {output_path}.")


def save_summary_to_latex(all_scores, output_path):
    """
    Save average performance by method to LaTeX file

    Args:
        all_scores: Dictionary of scores by method
        output_path: Save destination path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare data for DataFrame
    table_data = []

    for method, scores in all_scores.items():
        avg_acc = np.mean(scores["acc"])
        avg_nmi = np.mean(scores["nmi"])
        avg_ari = np.mean(scores["ari"])
        avg_f1 = np.mean(scores["f1"])
        count = len(scores["acc"])

        table_data.append(
            {
                "Method": method,
                "Avg ACC": f"{avg_acc:.4f}",
                "Avg NMI": f"{avg_nmi:.4f}",
                "Avg ARI": f"{avg_ari:.4f}",
                "Avg F1": f"{avg_f1:.4f}",
                "Num Datasets": count,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Save to LaTeX file
    latex_content = df.to_latex(
        index=False,
        escape=False,
        caption="Average clustering performance across all datasets.",
        label="tab:clustering_summary",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"LaTeX format summary saved to {output_path}.")


def compute_common_available_nodes(available_nodes_per_method, n_times):
    """
    Calculate node×time pairs commonly available across all methods

    Args:
        available_nodes_per_method: {method: {time: set of node_ids}}
        n_times: Number of time steps

    Returns:
        common_nodes: {time: set of node_ids} - Nodes common to all methods
    """
    if not available_nodes_per_method:
        return {}

    common_nodes = {}

    for t in range(n_times):
        # Collect node sets that each method has at this time
        time_node_sets = []
        for method_name, method_nodes in available_nodes_per_method.items():
            if t in method_nodes:
                time_node_sets.append(method_nodes[t])
            else:
                # If a method has no nodes at this time, use empty set
                time_node_sets.append(set())

        if time_node_sets:
            # Calculate intersection of all methods
            common_time_nodes = time_node_sets[0]
            for node_set in time_node_sets[1:]:
                common_time_nodes = common_time_nodes.intersection(node_set)

            if common_time_nodes:  # Record only when there are common nodes
                common_nodes[t] = common_time_nodes

    return common_nodes


def run_clustering_comparison(args):
    """
    Execute clustering comparison experiment for specified datasets and methods
    """
    print("=" * 80)
    print("Clustering Evaluation Experiment Started")
    print("=" * 80)

    # Dataset and cluster count settings
    k_dict = {
        "synthetic_1": 3,
        "synthetic_2": 3,
        "school": 9,
        "brain": 10,
        "stock": 11,
        "stock2": 11,
        "copenhagen_bt_daily": 7,
        "copenhagen_bt_hourly":7
    }

    # Dataset and node count settings
    n_nodes_dict = {
        "synthetic_1": 200,
        "synthetic_2": 200,
        "school": 327,
        "brain": 5000,
        "stock": 417,
        "stock2": 417,
        "copenhagen_bt_daily": 692,
        "copenhagen_bt_hourly": 692,
    }

    # Dataset list setup
    if args.dataset != "all":
        datasets = [args.dataset]
    else:
        datasets = list(k_dict.keys())

    # Save overall results
    all_results = []
    all_scores = {}  # {method: {metric: [score, ...]}}

    for dataset in datasets:
        print(f"\n{'='*20} Dataset: {dataset} {'='*20}")

        if dataset not in k_dict:
            print(
                f"Warning: Cluster count for dataset {dataset} is not defined. Skipping."
            )
            continue

        k = k_dict[dataset]
        label_path = f"../../data/{dataset}/node2label.txt"
        n_nodes = n_nodes_dict.get(dataset, None)

        print(f"Dataset: {dataset}")
        print(f"Cluster count: {k}")
        print(f"Label file: {label_path}")
        print(f"Node count: {n_nodes}")

        # Get number of time steps
        try:
            true_labels_per_time = custom_io.load_labels_per_time(label_path, n_nodes)
            n_times = len(true_labels_per_time)
            print(f"Time steps: {n_times}")
        except Exception as e:
            print(f"Warning: Failed to load labels: {e}")
            continue

        # Generate embedding file paths
        embedding_paths = find_embedding_files(dataset)

        if not embedding_paths:
            print(
                f"Warning: No embedding files found for dataset {dataset}."
            )
            continue

        print(f"Number of embedding files found: {len(embedding_paths)}")

        # Survey available nodes for each method
        print("Surveying available nodes...")
        available_nodes_per_method = survey_available_nodes_per_method(
            embedding_paths, n_nodes, n_times
        )

        # Display available node information
        for method_name in embedding_paths.keys():
            if method_name in available_nodes_per_method:
                total_available = sum(
                    len(nodes)
                    for nodes in available_nodes_per_method[method_name].values()
                )
                time_steps_available = len(available_nodes_per_method[method_name])
                print(
                    f"  {method_name}: {total_available} node×time pairs, {time_steps_available} time steps"
                )

        # Calculate nodes commonly available across all methods
        common_available_nodes = compute_common_available_nodes(
            available_nodes_per_method, n_times
        )
        print("\nNodes commonly available across all methods:")
        for t, nodes in common_available_nodes.items():
            print(f"  Time {t}: {len(nodes)} nodes")

        print("-" * 60)
        print(
            f"{'Method':<30} | {'ACC':<8} | {'NMI':<8} | {'ARI':<8} | "
            f"{'F1':<8} | {'Nodes':<8} | {'Clusters':<8} | {'TimeSteps':<8}"
        )
        print("-" * 100)

        # Evaluate each method
        dataset_results = []
        available_methods = []

        for method_name, emb_path in embedding_paths.items():
            # Use commonly available nodes across all methods
            result = test_external_score(
                emb_path, label_path, k, method_name, n_nodes, common_available_nodes
            )
            if result is not None:
                result["dataset"] = dataset
                dataset_results.append(result)
                all_results.append(result)
                available_methods.append(method_name)
                # Aggregate scores
                if method_name not in all_scores:
                    all_scores[method_name] = {
                        "acc": [],
                        "nmi": [],
                        "ari": [],
                        "f1": [],
                    }
                all_scores[method_name]["acc"].append(result["acc"])
                all_scores[method_name]["nmi"].append(result["nmi"])
                all_scores[method_name]["ari"].append(result["ari"])
                all_scores[method_name]["f1"].append(result["f1"])

        # Display best results for dataset
        if dataset_results:
            print("-" * 80)
            print(
                f"Methods with computed embeddings: "
                f"{len(available_methods)}/{len(embedding_paths)}"
            )
            # Sort methods in specified order
            sorted_methods = sort_methods(available_methods)
            print(f"Methods: {', '.join(sorted_methods)}")

            # Display best results for dataset
            best_acc = max(dataset_results, key=lambda x: x["acc"])
            best_nmi = max(dataset_results, key=lambda x: x["nmi"])
            best_ari = max(dataset_results, key=lambda x: x["ari"])
            best_f1 = max(dataset_results, key=lambda x: x["f1"])

            print(f"Best ACC: {best_acc['method']} ({best_acc['acc']:.4f})")
            print(f"Best NMI: {best_nmi['method']} ({best_nmi['nmi']:.4f})")
            print(f"Best ARI: {best_ari['method']} ({best_ari['ari']:.4f})")
            print(f"Best F1:  {best_f1['method']} ({best_f1['f1']:.4f})")
        else:
            print("Warning: No valid results for this dataset.")

    # Overall summary
    if all_results:
        print(f"\n{'='*20} Overall Summary {'='*20}")

        # Statistics of available methods
        unique_methods = set(result["method"] for result in all_results)
        unique_datasets = set(result["dataset"] for result in all_results)

        print(f"Number of evaluated methods: {len(unique_methods)}")
        print(f"Number of evaluated datasets: {len(unique_datasets)}")
        print(f"Total evaluations: {len(all_results)}")
        print()

        # Average performance by method
        print(
            f"{'Method':<30} | {'Avg ACC':<8} | {'Avg NMI':<8} | "
            f"{'Avg ARI':<8} | {'Avg F1':<8}"
        )
        print("-" * 90)
        method_list = sort_methods(list(all_scores.keys()))
        for i, method in enumerate(method_list):
            scores = all_scores[method]
            avg_acc = np.mean(scores["acc"])
            avg_nmi = np.mean(scores["nmi"])
            avg_ari = np.mean(scores["ari"])
            avg_f1 = np.mean(scores["f1"])
            count = len(scores["acc"])
            print(
                f"{method:<30} | {avg_acc:.4f}   | {avg_nmi:.4f}   | "
                f"{avg_ari:.4f}   | {avg_f1:.4f}   | {count}"
            )

        # Identify best methods overall (among methods with computed embeddings)
        best_overall = {}
        for metric in ["acc", "nmi", "ari", "f1"]:
            best_method = max(
                all_scores.keys(),
                key=lambda m: np.mean(all_scores[m][metric]),
            )
            best_score = np.mean(all_scores[best_method][metric])
            best_overall[metric] = (best_method, best_score)

        print("-" * 80)
        print("Overall Best Methods:")
        for metric, (method, score) in best_overall.items():
            print(f"  {metric.upper()}: {method} ({score:.4f})")

        # Method availability report
        print(f"\n{'='*20} Method Availability Report {'='*20}")
        total_possible = len(unique_datasets) * len(unique_methods)
        print(f"Theoretical maximum evaluations: {total_possible}")
        print(f"Actual evaluations: {len(all_results)}")
        print(f"Completion rate: {len(all_results)/total_possible*100:.1f}%")

        for method in sort_methods(list(unique_methods)):
            # Calculate actual number of results for each method
            method_results = [r for r in all_results if r["method"] == method]
            total_for_method = len(method_results)
            print(
                f"  {method}: {total_for_method}/{len(unique_datasets)} "
                f"datasets ({total_for_method/len(unique_datasets)*100:.1f}%)"
            )

        # Display results in table format
        format_result_table(all_results, all_scores)

        # Save results to CSV files
        if args.output:
            # Save detailed results
            detailed_output_path = args.output
            save_results_to_csv(all_results, detailed_output_path)

            # Save summary results
            summary_output_path = args.output.replace(".csv", "_summary.csv")
            save_summary_to_csv(all_scores, summary_output_path)

            # Save table format results
            table_output_path = args.output.replace(".csv", "_table.csv")
            save_table_to_csv(all_results, all_scores, table_output_path)

            # Save in LaTeX format
            latex_output_path = args.output.replace(".csv", "_table.tex")
            save_table_to_latex(all_results, all_scores, latex_output_path)

            summary_latex_path = args.output.replace(".csv", "_summary.tex")
            save_summary_to_latex(all_scores, summary_latex_path)
        else:
            # Default save destination
            timestamp = args.dataset if args.dataset != "all" else "all_datasets"
            detailed_output_path = f"result/clustering_results_{timestamp}.csv"
            summary_output_path = f"result/clustering_summary_{timestamp}.csv"
            table_output_path = f"result/clustering_table_{timestamp}.csv"
            latex_output_path = f"result/clustering_table_{timestamp}.tex"
            summary_latex_path = f"result/clustering_summary_{timestamp}.tex"

            save_results_to_csv(all_results, detailed_output_path)
            save_summary_to_csv(all_scores, summary_output_path)
            save_table_to_csv(all_results, all_scores, table_output_path)
            save_table_to_latex(all_results, all_scores, latex_output_path)
            save_summary_to_latex(all_scores, summary_latex_path)

    else:
        print("\nWarning: No evaluable results found.")
        print("Please check the following:")
        print("1. Are embedding files present in the correct path?")
        print("2. Does the label file exist?")
        print("3. Do the specified parameters match the generated file names?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clustering evaluation for temporal graph embeddings"
    )

    # Dataset specification
    parser.add_argument(
        "--dataset",
        type=str,
        default="brain",
        choices=[
            "synthetic_1",
            "synthetic_2",
            "school",
            "brain",
            "stock",
            "stock2",
            "copenhagen_bt_daily",
            "copenhagen_bt_hourly",
            "all",
        ],
        help="Dataset to evaluate ('all' for all datasets)",
    )

    # Result saving parameters
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as CSV file (if not specified, automatically saved to result/ directory)",
    )

    parser.add_argument("--verbose", action="store_true", help="Display detailed output")

    args = parser.parse_args()

    # Execute clustering comparison experiment
    run_clustering_comparison(args)
