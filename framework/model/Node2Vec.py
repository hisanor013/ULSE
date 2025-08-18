import os
import numpy as np
import random
import networkx as nx
from gensim.models import Word2Vec
import multiprocessing as mp
from .utils import load_edge_list_simple
from .base import BaseEmbedder


class Graph:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    next = cur_nbrs[
                        alias_draw(
                            alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1]
                        )
                    ]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(
                    self.node2vec_walk(walk_length=walk_length, start_node=node)
                )

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [
                G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))
            ]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def _compute_node2vec_for_time(args):
    """
    Function to compute Node2Vec embeddings for a single time step
    Helper function for parallel processing
    """
    (
        t_idx,
        original_time,
        edges_at_time,
        node_list,
        emb_size,
        walk_length,
        num_walks,
        window_size,
        p,
        q,
        workers,
        iter_count,
        weighted,
        directed,
    ) = args

    print(f"Computing Node2Vec embeddings for graph at time [{original_time}]...")

    # Build graph
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edges_at_time)
    if not weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

    # Node2Vec Graph object
    node2vec_graph = Graph(G, directed, p, q)
    node2vec_graph.preprocess_transition_probs()
    walks = node2vec_graph.simulate_walks(num_walks, walk_length)
    walks = [list(map(str, walk)) for walk in walks]

    model = Word2Vec(
        walks,
        vector_size=emb_size,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=iter_count,
    )

    # Store embeddings in node order
    embedding = np.zeros((len(node_list), emb_size), dtype=np.float32)
    for i, node in enumerate(node_list):
        if str(node) in model.wv:
            embedding[i, :] = model.wv[str(node)]
        else:
            embedding[i, :] = 0.0

    print(f"  Time {original_time} embedding shape: {embedding.shape}")
    return t_idx, embedding


class Node2Vec(BaseEmbedder):
    """
    Class to compute Node2Vec embeddings for each time step
    Inherits from BaseEmbedder to provide a unified interface
    Supports parallel processing
    """

    def __init__(
        self,
        edge_path,
        emb_size,
        output_path,
        walk_length=80,
        num_walks=10,
        window_size=10,
        p=1.0,
        q=1.0,
        workers=8,
        iter=1,
        weighted=False,
        directed=False,
        n_jobs=None,
    ):
        super().__init__(edge_path, emb_size, output_path)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.workers = workers
        self.iter = iter
        self.weighted = weighted
        self.directed = directed
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()

        # Node2Vec-specific attributes
        self.node_list = None
        self.time_list = None
        self.node_map = None
        self.time_map = None

    def compute_embedding(self):
        """
        Compute Node2Vec embeddings for each time step (with parallel processing support)
        Implemented according to BaseEmbedder interface
        """
        print(
            "=== Node2Vec: Starting embedding computation for each time step (parallel processing) ==="
        )
        print(f"Parallelism: {self.n_jobs}")

        # Load edge list
        edges, node_list, node_map, time_map = load_edge_list_simple(self.edge_path)
        self.node_list = node_list
        self.node_map = node_map
        self.time_map = time_map
        # time_map keys are original time values (float), values are integer indices
        self.time_list = sorted(list(time_map.keys()))
        N = len(node_list)
        D = self.emb_size
        T = len(self.time_list)

        # Split edges by time (grouped by integer indices)
        edges_by_time = {idx: [] for idx in range(T)}
        for s, t, time_idx in edges:
            edges_by_time[time_idx].append((s, t))

        # Prepare arguments for parallel processing
        parallel_args = []
        for t_idx in range(T):
            original_time = self.time_list[t_idx]
            args = (
                t_idx,
                original_time,
                edges_by_time[t_idx],
                node_list,
                D,
                self.walk_length,
                self.num_walks,
                self.window_size,
                self.p,
                self.q,
                self.workers,
                self.iter,
                self.weighted,
                self.directed,
            )
            parallel_args.append(args)

        # Compute embeddings for each time step using parallel processing
        all_embs = np.zeros((T, N, D), dtype=np.float32)

        if self.n_jobs == 1:
            # Single process execution
            for args in parallel_args:
                t_idx, embedding = _compute_node2vec_for_time(args)
                all_embs[t_idx] = embedding
        else:
            # Multi-process execution
            with mp.Pool(processes=self.n_jobs) as pool:
                results = pool.map(_compute_node2vec_for_time, parallel_args)

                # Store results in array
                for t_idx, embedding in results:
                    all_embs[t_idx] = embedding

        # Set to BaseEmbedder's embedding attribute
        self.embedding = all_embs
        print("=== Node2Vec: Embedding computation for all time steps completed ===")

    def save_node_embeddings(self, output_path=None):
        """
        Save embeddings for all time steps to a single file
        Implemented according to BaseEmbedder interface
        Save format:
        <node_count> <dimension>
        node_id emb1 emb2 ... embD  # t=0
        ...
        node_id emb1 emb2 ... embD  # t=1
        ...
        """
        if output_path is None:
            output_path = self.output_path
        if self.embedding is None:
            raise ValueError(
                "Embeddings are not computed. Please run compute_embedding() first."
            )

        N, D, T = len(self.node_list), self.emb_size, len(self.time_list)
        print(f"Nodes: {N}, Time steps: {T}, Embedding dimension: {D}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(f"{N} {D}\n")
            for t_idx, t in enumerate(self.time_list):
                for i, node in enumerate(self.node_list):
                    emb = self.embedding[t_idx, i]
                    emb_str = " ".join(map(str, emb))
                    f.write(f"{node} {emb_str}\n")

        print(f"Node2Vec all-time embeddings saved successfully: {output_path}")

    def get_embedding_statistics(self):
        """
        Get embedding statistics
        """
        if self.embedding is None:
            return {"status": "not_computed"}
        N, D, T = self.embedding.shape
        return {
            "method": "Node2Vec",
            "embedding_shape": (N, D, T),
            "node_count": N,
            "time_steps": T,
            "embedding_dimension": D,
        }

    @staticmethod
    def create_edgelist_from_temporal_data(data_path, output_path):
        """
        Static method to convert temporal data to edge list
        """
        print(f"Converting temporal data to edge list: {data_path} → {output_path}")
        with open(data_path, "r") as infile:
            with open(output_path, "w") as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        outfile.write(f"{parts[0]} {parts[1]}\n")
        print("Edge list conversion completed")
