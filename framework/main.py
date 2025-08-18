# -*- coding: utf-8 -*-
import datetime
import argparse
import os
import tempfile

from model.USE import USE
from model.TemporalCut import TemporalCut
from model.Node2Vec import Node2Vec
from model.OMNI import OMNI
from model.utils import load_edge_list_simple


def apply_time_binning_to_args(args, edge_path):
    """
    Apply time binning to the dataset's adjacency matrix list, create a temporary file with binned data,
    and set the new edge list path to args.
    """
    edges, node_list, node_map, time_map = load_edge_list_simple(edge_path)
    T = len(time_map)
    max_timesteps = args.max_timesteps if hasattr(args, "max_timesteps") else 1000
    use_time_binning = (
        args.use_time_binning if hasattr(args, "use_time_binning") else True
    )
    if not use_time_binning or T <= max_timesteps:
        return edge_path  # Use as is

    print(f"Before binning: {T} timesteps, After binning: {max_timesteps} timesteps")
    bin_size = max(1, T // max_timesteps)
    new_T = (T + bin_size - 1) // bin_size
    print(f"Bin size: {bin_size}, Actual timesteps after binning: {new_T}")

    # Aggregate edges for each bin
    new_edges = []
    for t in range(new_T):
        start_idx = t * bin_size
        end_idx = min((t + 1) * bin_size, T)
        # Extract edges within the bin
        bin_edges = [e for e in edges if start_idx <= e[2] < end_idx]
        # Replace timesteps within the bin with bin number
        for s, d, _ in bin_edges:
            new_edges.append((s, d, t))

    # Save to temporary file
    tmp_dir = tempfile.mkdtemp()
    new_edge_path = f"{tmp_dir}/binned_{args.dataset}.txt"
    with open(new_edge_path, "w") as f:
        for s, d, t in new_edges:
            f.write(f"{s} {d} {t}\n")
    print(f"Saved binned edge list to temporary file: {new_edge_path}")
    return new_edge_path


def main_train(args):
    # Specify edge path
    edge_path = f"../data/{args.dataset}/{args.dataset}.txt"
    emb_size = args.emb_size

    print(f"Using edge list: {edge_path}")

    # Execute binning only once here
    edge_path = apply_time_binning_to_args(args, edge_path)

    # Configure method list
    if args.method != "all":
        methods = [args.method]
    else:
        methods = ["USE", "TemporalCut", "Node2Vec", "OMNI"]

    total_start = datetime.datetime.now()
    for method in methods:
        print(f"\n==== Starting {method} execution ====")
        method_start = datetime.datetime.now()
        run_method(method, args, edge_path, emb_size)
        method_end = datetime.datetime.now()
        method_time = method_end - method_start
        print(f"==== {method} execution completed: {method_time} ====")

    total_end = datetime.datetime.now()
    total_time = total_end - total_start
    print(f"\n=== Total execution time: {total_time} ===")


def run_method(method, args, edge_path, emb_size):
    """
    Common processing to execute the specified method

    Args:
        method: Method name
        args: Command line arguments
        edge_path: Edge list path
        emb_size: Embedding dimension
    """
    if method == "USE":
        run_use(args, edge_path, emb_size)
    elif method == "TemporalCut":
        run_temporal_cut(args, edge_path, emb_size)
    elif method == "Node2Vec":
        embedder = run_node2vec(args, edge_path, emb_size)
        run_embedder(embedder, None, "Node2Vec")
    elif method == "OMNI":
        embedder = run_omni(args, edge_path, emb_size)
        run_embedder(embedder, None, "OMNI")
    else:
        raise ValueError(f"Unknown method: {method}")


def run_use(args, edge_path, emb_size):
    """Execute USE method"""
    # Execute USE (ULSE/UASE) - execute both rep_types when method is all
    if args.method == "all":
        rep_types = ["UASE", "ULSE-n1", "ULSE-n2"]
    else:
        rep_types = [args.rep_type]

    for rep_type in rep_types:
        print(f"  USE rep_type: {rep_type}")
        output_path = f"../emb/{args.dataset}/" f"{args.dataset}_USE_{rep_type}.emb"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        use_start = datetime.datetime.now()
        embedder = USE(edge_path, emb_size, output_path, rep_type=rep_type)
        run_embedder(embedder, output_path, "USE")
        use_end = datetime.datetime.now()
        use_time = use_end - use_start
        print(f"  USE {rep_type} execution time: {use_time}")


def run_temporal_cut(args, edge_path, emb_size):
    """Execute TemporalCut method"""
    # Execute TemporalCut - execute both cut_types when method is all
    if args.method == "all":
        cut_types = ["sparse", "normalized"]
    else:
        cut_types = [args.cut_type]

    for cut_type in cut_types:
        print(f"  TemporalCut cut_type: {cut_type}")
        output_path = (
            f"../emb/{args.dataset}/"
            f"{args.dataset}_TemporalCut_"
            f"{args.temporal_method}_{cut_type}.emb"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("TemporalCut settings:")
        print(f"    Method: {args.temporal_method}")
        print(f"    Cut type: {cut_type}")
        print(f"    Swap cost: {args.beta}")
        print(f"    Rank (for fast): {args.rank_r}")
        print(f"    Max iterations: {args.max_iter}")
        print(f"    Convergence threshold: {args.tolerance}")

        temporal_cut_start = datetime.datetime.now()
        embedder = TemporalCut(
            edge_path=edge_path,
            emb_size=emb_size,
            output_path=output_path,
            method=args.temporal_method,
            cut_type=cut_type,
            beta=args.beta,
            rank_r=args.rank_r,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            n_jobs=args.n_jobs,
            use_sparse=args.use_sparse,
            force_exact_computation=args.force_exact_computation,
        )
        run_embedder(embedder, output_path, "TemporalCut")
        temporal_cut_end = datetime.datetime.now()
        temporal_cut_time = temporal_cut_end - temporal_cut_start
        print(f"  TemporalCut {cut_type} execution time: {temporal_cut_time}")


def run_node2vec(args, edge_path, emb_size):
    """Execute Node2Vec method"""
    output_path = f"../emb/{args.dataset}/" f"{args.dataset}_Node2Vec.emb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Node2Vec settings:")
    print(f"  Walk length: {args.walk_length}")
    print(f"  Number of walks: {args.num_walks}")
    print(f"  p parameter: {args.p}")
    print(f"  q parameter: {args.q}")
    print(f"  Window size: {args.window_size}")
    print(f"  Parallelism: {args.n_jobs}")

    node2vec_start = datetime.datetime.now()
    embedder = Node2Vec(
        edge_path=edge_path,
        emb_size=emb_size,
        output_path=output_path,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window_size=args.window_size,
        p=args.p,
        q=args.q,
        workers=args.workers,
        iter=args.iter,
        weighted=args.weighted,
        directed=args.directed,
        n_jobs=args.n_jobs,
    )
    run_embedder(embedder, output_path, "Node2Vec")
    node2vec_end = datetime.datetime.now()
    node2vec_time = node2vec_end - node2vec_start
    print(f"  Node2Vec execution time: {node2vec_time}")
    return embedder


def run_omni(args, edge_path, emb_size):
    """Execute OMNI method"""
    output_path = f"../emb/{args.dataset}/" f"{args.dataset}_OMNI.emb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("OMNI settings (low-rank approximation version):")
    print(f"  Embedding dimension: {emb_size}")
    print(f"  Parallel jobs: {args.n_jobs}")
    print(f"  Use sparse matrix: {args.use_sparse}")
    print(f"  Low-rank approximation factor: {args.low_rank_factor}")

    omni_start = datetime.datetime.now()
    embedder = OMNI(
        edge_path=edge_path,
        emb_size=emb_size,
        output_path=output_path,
        n_jobs=args.n_jobs,
        use_sparse=args.use_sparse,
        low_rank_factor=args.low_rank_factor,
    )
    run_embedder(embedder, output_path, "OMNI")
    omni_end = datetime.datetime.now()
    omni_time = omni_end - omni_start
    print(f"  OMNI execution time: {omni_time}")
    return embedder


def run_embedder(embedder, output_path, method_name):
    """
    Execute common processing for embedding classes

    Args:
        embedder: Instance of BaseEmbedder
        output_path: Output path (use embedder.output_path if None)
        method_name: Method name
    """
    # Common processing: compute and save embeddings
    embedder.compute_embedding()
    if output_path:
        embedder.save_node_embeddings(output_path)
    else:
        embedder.save_node_embeddings()

    # Display statistics
    stats = embedder.get_embedding_statistics()
    print(f"{method_name} statistics:")
    print(stats)


if __name__ == "__main__":
    data = "brain"
    k_dict = {
        "synthetic_1": 3,
        "synthetic_2": 3,
        "school": 9,
        "brain": 10,
        "stock": 11,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=data)
    parser.add_argument(
        "--method",
        type=str,
        default="USE",
        choices=["USE", "TemporalCut", "Node2Vec", "OMNI", "all"],
        help="Embedding method to use (use 'all' to run all methods)",
    )

    # USE parameters
    parser.add_argument(
        "--rep_type",
        type=str,
        default="ULSE-n1",
        choices=["UASE", "ULSE-n1", "ULSE-n2"],
        help="Representation type for USE (ULSE/UASE)",
    )

    # Node2Vec parameters
    parser.add_argument(
        "--walk_length",
        type=int,
        default=80,
        help="Length of walk per source for Node2Vec",
    )
    parser.add_argument(
        "--num_walks",
        type=int,
        default=10,
        help="Number of walks per source for Node2Vec",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Context size for Word2Vec optimization in Node2Vec",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1.0,
        help="Return hyperparameter for Node2Vec",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="In-out hyperparameter for Node2Vec",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for Node2Vec",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="Number of epochs in SGD for Node2Vec",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Boolean specifying weighted graph for Node2Vec",
    )

    # TemporalCut parameters
    parser.add_argument(
        "--temporal_method",
        type=str,
        default="fast",
        choices=["diff", "laplacian", "prod", "fast"],
        help=(
            "TemporalCut algorithm: diff (fastest), laplacian (standard), "
            "prod (highest quality), fast (large-scale)"
        ),
    )
    parser.add_argument(
        "--cut_type",
        type=str,
        default="sparse",
        choices=["sparse", "normalized"],
        help="Cut type: sparse (node count) or normalized (volume-based)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Temporal swap cost (higher = more temporal consistency)",
    )
    parser.add_argument(
        "--rank_r",
        type=int,
        default=32,
        help="Low-rank approximation for fast_cut method",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum iterations for power method",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance for iterative methods",
    )

    # OMNI parameters
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (used by Node2Vec and OMNI)",
    )
    parser.add_argument(
        "--use_sparse",
        type=bool,
        default=True,
        help="Whether to use sparse matrices (default: True)",
    )
    parser.add_argument(
        "--low_rank_factor",
        type=float,
        default=2.0,
        help="Low-rank approximation factor (emb_size * low_rank_factor)",
    )
    parser.add_argument(
        "--force_exact_computation",
        action="store_true",
        default=False,
        help="Disable approximate computation for result consistency (default: False)",
    )

    # Time binning parameters
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=100,
        help="Maximum number of timesteps (use bin division when exceeded)",
    )
    parser.add_argument(
        "--use_time_binning",
        action="store_true",
        default=False,
        help="Whether to use time binning when there are many timesteps",
    )

    # Common parameters
    parser.add_argument("--emb_size", type=int, default=k_dict[data])
    args = parser.parse_args()

    # If emb_size is not specified on command line, set automatically based on dataset
    import sys

    if not any(arg.startswith("--emb_size") for arg in sys.argv):
        if args.dataset in k_dict:
            args.emb_size = k_dict[args.dataset]
        else:
            print(
                f"Warning: No embedding dimension found in k_dict for dataset {args.dataset}. Using emb_size={args.emb_size}."
            )

    main_train(args)
