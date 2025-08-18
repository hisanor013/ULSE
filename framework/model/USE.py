from .base import BaseEmbedder
import numpy as np
from .utils import load_edge_list_simple, n1_laplacian_list, n2_laplacian_list
from typing import Union, Callable, List


class USE(BaseEmbedder):
    def __init__(
        self, edge_path, emb_size, output_path, rep_type="ULSE-n1", regularized=0.1
    ):
        super().__init__(edge_path, emb_size, output_path)
        self.rep_type = rep_type
        print(f"USE initialization completed. rep_type: {rep_type}")
        print("Loading edge list.")
        self.A_list, self.node_num, self.node_list = self.load_adjacency_list()
        print(
            "Edge list loading and adjacency matrix creation completed. Node count:",
            self.node_num,
        )
        self.regularized = regularized

    def load_adjacency_list(self):
        edges, node_list, node_map, time_map = load_edge_list_simple(self.edge_path)
        print("Creating adjacency matrix list.")
        T = len(time_map)
        node_num = len(node_list)
        A_list = [np.zeros((node_num, node_num)) for _ in range(T)]

        print("Adjacency matrix creation information:")
        print(f"  Time steps: {T}")
        print(f"  Node count: {node_num}")
        print(f"  Edge count: {len(edges)}")
        print(f"  First 5 edges: {edges[:5]}")

        for s, t, time in edges:
            i, j = node_map[s], node_map[t]
            t_idx = time
            if t_idx >= T:
                print(f"Error: time index {t_idx} exceeds time count {T}")
                raise IndexError(f"Time index {t_idx} is out of range")
            A_list[t_idx][i, j] = 1
            A_list[t_idx][j, i] = 1

        return A_list, node_num, node_list

    def unfold(self, A_list):
        return np.hstack(A_list)

    def compute_embedding(self):
        print("Starting embedding computation.")
        rep_type = self.rep_type
        A_list = self.A_list
        T = len(A_list)

        if rep_type == "UASE":
            rep_list = A_list
        elif rep_type == "ULSE-n1":
            rep_list = n1_laplacian_list(A_list, regularized=self.regularized)
        elif rep_type == "ULSE-n2":
            rep_list = n2_laplacian_list(A_list, regularized=self.regularized)
        else:
            raise ValueError(f"Unknown rep_type: {rep_type}")

        unfolded_rep = self.unfold(rep_list)
        U, s, Vt = np.linalg.svd(unfolded_rep, full_matrices=False)

        Sigma_sqrt = np.diag(np.sqrt(s))
        eps = 1e-10
        has_zero_singular = np.any(s < eps)
        if has_zero_singular:
            print(
                "⚠️ USE: Some singular values are zero or near zero. Inverse square root replaced with 0."
            )
        s_inv_sqrt = np.array([1 / np.sqrt(val) if val > eps else 0.0 for val in s])
        Sigma_sqrt_inv = np.diag(s_inv_sqrt)

        if rep_type == "UASE":
            Y = Vt.T @ Sigma_sqrt
        elif rep_type == "ULSE-n1":
            Y = Vt.T @ Sigma_sqrt - np.tile(U @ Sigma_sqrt_inv, (T, 1))
        elif rep_type == "ULSE-n2":
            Y = Vt.T @ Sigma_sqrt
        else:
            raise ValueError(f"Unknown rep_type: {rep_type}")

        # Selection of singular vectors to use
        if rep_type == "UASE":
            self.embedding = Y[:, : self.emb_size]  # From largest
        elif rep_type == "ULSE-n1":
            # Exclude the smallest singular value's singular vector, use 2nd to (k+1)th
            self.embedding = Y[:, -(self.emb_size + 1) : -1]
        else:  # rep_type == "ULSE-n2"
            self.embedding = Y[:, : self.emb_size]  # From largest

        print("Embedding computation completed.")

    def save_node_embeddings(self, path=None):
        print("Starting node embedding save.")
        if self.embedding is None:
            raise ValueError(
                "Embedding has not been computed. Please call compute_embedding() first."
            )
        if path is None:
            path = self.output_path

        nT, d = self.embedding.shape
        n = self.node_num
        T = nT // n
        print(f"Node count: {n}, Time steps: {T}, Embedding dimension: {d}")

        with open(path, "w") as f:
            f.write(f"{len(self.node_list)} {d}\n")
            for t in range(T):
                for node_idx, node in enumerate(self.node_list):
                    emb_str = " ".join(map(str, self.embedding[t * n + node_idx]))
                    f.write(f"{node} {emb_str}\n")

        print("Node embedding save completed. File: ", path)

    def get_embedding_statistics(self):
        """
        Get embedding statistics
        Implemented following the BaseEmbedder interface

        Returns:
            dict: Dictionary of statistical information
        """
        if self.embedding is None:
            return {"status": "not_computed"}

        # Utilize the default implementation of BaseEmbedder
        stats = super().get_embedding_statistics()

        # Add USE-specific information
        nT, d = self.embedding.shape
        n = self.node_num
        T = nT // n

        stats.update(
            {
                "rep_type": self.rep_type,
                "embedding_shape": (n, d, T),
                "node_count": n,
                "time_steps": T,
                "embedding_dimension": d,
            }
        )

        return stats
