import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .utils import load_edge_list_simple
from .base import BaseEmbedder


class OMNI(BaseEmbedder):

    def __init__(
        self,
        edge_path: str,
        emb_size: int,
        output_path: str,
        n_jobs: int = None,
        use_sparse: bool = True,
        low_rank_factor: float = 2.0,
    ):
        """
        Initialize OMNI

        Args:
            edge_path: Path to edge list file
            emb_size: Dimension of embeddings
            output_path: Output path
            n_jobs: Number of parallel jobs (CPU count if None)
            use_sparse: Whether to use sparse matrices
            low_rank_factor: Low-rank approximation factor (emb_size * low_rank_factor)
        """
        super().__init__(edge_path, emb_size, output_path)

        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        self.use_sparse = use_sparse
        self.low_rank_factor = low_rank_factor
        self.node_list = None
        self.time_list = None
        self.node_map = None
        self.time_map = None
        self.node_num = 0
        self.time_steps = 0

    def _low_rank_approximation(self, A: np.ndarray, rank: int) -> tuple:
        """
        Compute low-rank approximation

        Args:
            A: Input matrix
            rank: Approximation rank

        Returns:
            (U, S, Vt): Low-rank decomposition result
        """
        # Use full SVD if rank is too large
        if rank >= min(A.shape):
            rank = min(A.shape) - 1

        if self.use_sparse:
            A_sparse = csr_matrix(A)
            U, S, Vt = svds(A_sparse, k=rank, which="LM")
        else:
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

        return U, S, Vt

    def _compute_parallel_low_rank_approximations(
        self, As: list, target_rank: int
    ) -> list:
        """
        Compute low-rank approximations for each time step in parallel

        Args:
            As: List of adjacency matrices
            target_rank: Target rank

        Returns:
            low_rank_As: List of low-rank decomposition results
        """
        T = len(As)
        print(
            f"Starting low-rank approximation for each time step: rank {target_rank}, {self.n_jobs} parallel"
        )

        def compute_low_rank_approximation(t):
            """Low-rank approximation function for parallel processing"""
            U, S, Vt = self._low_rank_approximation(As[t], target_rank)
            return t, (U, S, Vt)

        # Execute low-rank approximation in parallel
        low_rank_As = [None] * T
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(compute_low_rank_approximation, t) for t in range(T)
            ]

            for future in futures:
                t, low_rank_result = future.result()
                low_rank_As[t] = low_rank_result

        print(f"Low-rank approximation completed: {T} time steps")
        return low_rank_As

    def _compute_efficient_omnibus_embedding(
        self, low_rank_As: list, K: int
    ) -> np.ndarray:
        """
        Efficient Omnibus embedding computation leveraging low-rank format
        Process in low-rank format without constructing large matrices

        Args:
            low_rank_As: List of low-rank decomposition results
            K: Embedding dimension

        Returns:
            XAs: Embedding results for each time step (T, n, K)
        """
        T = len(low_rank_As)
        n = low_rank_As[0][0].shape[0]  # Number of rows in U
        rank = low_rank_As[0][1].shape[0]  # Length of S

        print(
            f"Starting embedding computation: {T} time steps, {n} nodes, {rank} rank, {K} dimensions"
        )

        # Create combined matrix of low-rank representations for each time step
        U_combined = np.zeros((T * n, T * rank))
        S_combined = np.zeros(T * rank)

        for t in range(T):
            U, S, Vt = low_rank_As[t]
            start_idx = t * n
            start_rank = t * rank

            # Store U * S in combined matrix
            U_combined[start_idx : start_idx + n, start_rank : start_rank + rank] = (
                U * S.reshape(1, -1)
            )
            S_combined[start_rank : start_rank + rank] = S

        # SVD of combined low-rank matrix
        print(f"SVD of combined low-rank matrix: size {U_combined.shape}")

        if self.use_sparse and U_combined.shape[0] > 1000:
            U_combined_sparse = csr_matrix(U_combined)
            UA, SA, _ = svds(U_combined_sparse, k=K, which="LM")
        else:
            UA, SA, _ = np.linalg.svd(U_combined, full_matrices=False)
            UA, SA = UA[:, :K], SA[:K]

        # Split embedding results by time step
        XAs = np.zeros((T, n, K))
        for t in range(T):
            XAs[t] = UA[t * n : (t + 1) * n, :K] * np.sqrt(SA[:K])

        print("Efficient embedding computation completed")
        return XAs

    def _omni_embed_matrices(self, As: list, K: int) -> np.ndarray:
        """
        Efficient OMNI embedding computation using low-rank approximation

        Args:
            As: List of adjacency matrices
            K: Embedding dimension

        Returns:
            XAs: Embedding results for each time step (T, n, K)
        """
        T = len(As)
        n = As[0].shape[0]

        # Determine rank for low-rank approximation
        target_rank = min(int(K * self.low_rank_factor), n)
        target_rank = max(target_rank, K)  # At least K

        print(
            f"Starting efficient OMNI computation: {T} time steps, {n} nodes, {K} dimensions"
        )
        print(f"Low-rank approximation rank: {target_rank}")

        # Compute low-rank approximations for each time step in parallel
        low_rank_As = self._compute_parallel_low_rank_approximations(As, target_rank)

        # Compute efficient Omnibus embedding in low-rank format
        print("Starting efficient embedding computation in low-rank format")
        XAs = self._compute_efficient_omnibus_embedding(low_rank_As, K)

        return XAs

    def _load_and_preprocess_data(self):
        """
        Data loading and preprocessing with parallel processing

        Returns:
            As: List of adjacency matrices
        """
        print("Loading edge list...")

        # Load edge list
        edges, node_list, node_map, time_map = load_edge_list_simple(self.edge_path)
        print(
            f"Loading completed: {len(edges)} edges, {len(node_list)} nodes, {len(time_map)} time steps"
        )

        self.node_list = node_list
        self.node_map = node_map
        self.time_map = time_map
        self.node_num = len(node_list)
        self.time_steps = len(time_map)
        self.time_list = sorted(list(time_map.keys()))

        # Build adjacency matrices in parallel
        print("Starting parallel adjacency matrix construction...")

        def build_adjacency_matrix(t):
            A_t = np.zeros((self.node_num, self.node_num))
            for s, d, time in edges:
                if time == t:  # time is already converted to index
                    i, j = node_map[s], node_map[d]
                    A_t[i, j] = 1.0
                    A_t[j, i] = 1.0  # undirected graph
            return t, A_t

        # Parallel processing
        As = [None] * self.time_steps
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(build_adjacency_matrix, t)
                for t in range(self.time_steps)
            ]

            for future in futures:
                t, A_t = future.result()
                As[t] = A_t

        print(
            f"Adjacency matrix construction completed: {self.node_num} nodes × {self.time_steps} time steps"
        )
        return As

    def compute_embedding(self):
        """
        Compute parallel OMNI embedding
        Implemented according to BaseEmbedder interface
        """
        print("=== Starting OMNI embedding computation ===")
        print(f"Number of parallel jobs: {self.n_jobs}")

        # Parallel data loading and preprocessing
        As = self._load_and_preprocess_data()

        # OMNI embedding computation
        print("Computing OMNI embedding...")
        XAs = self._omni_embed_matrices(As, self.emb_size)

        # Set to BaseEmbedder's embedding attribute
        T, n, d = XAs.shape
        self.embedding = XAs.reshape(T * n, d)

        print(
            f"OMNI embedding computation completed! Final embedding shape: {self.embedding.shape}"
        )

    def save_node_embeddings(self, output_path=None):
        """
        Save node embeddings to file

        Args:
            output_path: Output path (use self.output_path if None)
        """
        if output_path is None:
            output_path = self.output_path
        if self.embedding is None:
            raise RuntimeError("Please call compute_embedding() first")

        nT, d = self.embedding.shape
        n = self.node_num
        T = nT // n

        print(f"Saving parallel embeddings: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Parallel saving process
        def save_time_step(t):
            lines = []
            for v, orig_node_id in enumerate(self.node_list):
                emb_idx = t * n + v
                emb_values = " ".join(map(str, self.embedding[emb_idx]))
                lines.append(f"{orig_node_id} {emb_values}")
            return t, lines

        with open(output_path, "w") as f:
            f.write(f"{n} {d}\n")
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(save_time_step, t) for t in range(T)]
                for future in futures:
                    t, lines = future.result()
                    for line in lines:
                        f.write(line + "\n")

        print(f"Parallel saving completed: {n} nodes × {T} time steps × {d} dimensions")

    def get_embedding_statistics(self):
        """
        Get embedding statistics

        Returns:
            dict: Dictionary of statistics
        """
        if self.embedding is None:
            return {"status": "not_computed"}

        # Leverage BaseEmbedder's default implementation
        stats = super().get_embedding_statistics()

        # Add OMNI-specific information
        nT, d = self.embedding.shape
        n = self.node_num
        T = nT // n

        stats.update(
            {
                "embedding_shape": (n, d, T),
                "node_count": n,
                "time_steps": T,
                "parallel_jobs": self.n_jobs,
                "use_sparse": self.use_sparse,
                "low_rank_factor": self.low_rank_factor,
            }
        )

        return stats
