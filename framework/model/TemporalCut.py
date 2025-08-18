import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import networkx as nx
from typing import List, Tuple, Dict
from .utils import unnormalized_laplacian, n1_laplacian, load_edge_list_simple
from .base import BaseEmbedder
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


class TemporalGraph:
    """
    Class for managing temporal graphs
    Holds multiple snapshots and represents time-series connections of the same nodes
    """

    def __init__(self, swap_cost: float = 1.0):
        self.graphs: List[nx.Graph] = []
        self.swap_cost = swap_cost
        self._node_list: List = []
        self._number_of_nodes: int = 0
        self._number_of_nodes_snap: List[int] = []

    def add_snapshot(self, adjacency_matrix: sp.spmatrix, node_list: List[str] = None):
        n = adjacency_matrix.shape[0]
        if node_list is None:
            node_list = list(range(n))
        G = nx.from_scipy_sparse_array(adjacency_matrix)
        for i, j in zip(*adjacency_matrix.nonzero()):
            if G.has_edge(i, j):
                G[i][j]["weight"] = adjacency_matrix[i, j]
        self.graphs.append(G)
        self._number_of_nodes_snap.append(n)
        if len(self.graphs) == 1:
            self._node_list = node_list
            self._number_of_nodes = n

    def num_snapshots(self) -> int:
        return len(self.graphs)

    def size(self) -> int:
        return self._number_of_nodes

    def get_snapshot(self, t: int) -> nx.Graph:
        return self.graphs[t]

    def nodes(self) -> List:
        return self._node_list

    def create_laplacian_matrix(self, normalized: bool = False) -> sp.spmatrix:
        """
        Construct temporal Laplacian matrix L
        Place each snapshot's L_t on block diagonal,
        Add connections between time steps with swap_cost
        Apply D^-1/2 L D^-1/2 in normalized mode
        """
        T = self.num_snapshots()
        n = self.size()
        total = n * T

        # Parallelized Laplacian matrix construction
        laplacians = self._compute_laplacians_parallel(normalized)

        # Add temporal swap connections
        L = self._add_temporal_connections(laplacians, T, n, total)

        if normalized:
            deg = np.array(L.sum(axis=1)).flatten()
            inv_sqrt = np.zeros_like(deg)
            nz = deg > 0
            inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
            D = sp.diags(inv_sqrt)
            L = D @ L @ D
        return L

    def _compute_laplacians_parallel(self, normalized: bool) -> List[sp.spmatrix]:
        """Parallelized Laplacian matrix computation"""
        T = self.num_snapshots()

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            laplacian_futures = []
            for t in range(T):
                future = executor.submit(self._compute_single_laplacian, t, normalized)
                laplacian_futures.append(future)

            # Collect results
            laplacians = [future.result() for future in laplacian_futures]

        return laplacians

    def _compute_single_laplacian(self, t: int, normalized: bool) -> sp.spmatrix:
        """Compute single Laplacian matrix (for parallelization)"""
        A = nx.adjacency_matrix(
            self.graphs[t], nodelist=self._node_list, weight="weight"
        )
        if normalized:
            return n1_laplacian(A)
        else:
            return unnormalized_laplacian(A)

    def _add_temporal_connections(
        self, laplacians: List[sp.spmatrix], T: int, n: int, total: int
    ) -> sp.spmatrix:
        """Add temporal swap connections"""
        rows, cols, vals = [], [], []

        # Place each time step's Laplacian on block diagonal
        for t, L_t in enumerate(laplacians):
            coo = L_t.tocoo()
            for i, j, v in zip(coo.row, coo.col, coo.data):
                rows.append(t * n + i)
                cols.append(t * n + j)
                vals.append(v)

        # Temporal swap
        for t in range(T - 1):
            for v in range(n):
                i1 = t * n + v
                i2 = (t + 1) * n + v
                rows += [i1, i2, i1, i2]
                cols += [i2, i1, i1, i2]
                vals += [
                    -self.swap_cost,
                    -self.swap_cost,
                    self.swap_cost,
                    self.swap_cost,
                ]

        return sp.coo_matrix((vals, (rows, cols)), shape=(total, total)).tocsr()

    def create_complete_laplacian_matrix(self) -> sp.spmatrix:
        """
        Construct L for complete graph on block diagonal for each snapshot
        """
        T = self.num_snapshots()
        n = self.size()
        total = n * T
        rows, cols, vals = [], [], []
        for t in range(T):
            off = t * n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        rows.append(off + i)
                        cols.append(off + i)
                        vals.append(n - 1.0)
                    else:
                        rows.append(off + i)
                        cols.append(off + j)
                        vals.append(-1.0)
        C = sp.coo_matrix((vals, (rows, cols)), shape=(total, total)).tocsr()
        return C


class TemporalCut(BaseEmbedder):
    """
    Spectral cut embedding for temporal graphs
    """

    def __init__(
        self,
        edge_path: str,
        emb_size: int,
        output_path: str,
        method: str = "diff",
        cut_type: str = "sparse",
        beta: float = 1.0,
        auto_adjust_beta: bool = True,
        rank_r: int = 32,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        # Parallelization parameters
        n_jobs: int = None,
        use_sparse: bool = True,
        # Parameters to ensure result consistency
        force_exact_computation: bool = False,
    ):
        super().__init__(edge_path, emb_size, output_path)
        if method not in {"diff", "laplacian", "prod", "fast"}:
            raise ValueError("Invalid method")
        if cut_type not in {"sparse", "normalized"}:
            raise ValueError("Invalid cut_type")
        self.method = method
        self.cut_type = cut_type
        self.beta = float(beta)
        self.auto_adjust_beta = auto_adjust_beta
        self.rank_r = int(rank_r)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.temporal_graph = None
        self._eigenvalues = None
        self._eigenvectors = None

        # Parallelization settings
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        self.use_sparse = use_sparse

        # Settings to ensure result consistency
        self.force_exact_computation = force_exact_computation

        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        edges, node_list, node_map, time_map = load_edge_list_simple(self.edge_path)
        n = len(node_list)
        T = len(time_map)

        print(f"Data loading complete: {n} nodes, {T} time steps, {len(edges)} edges")

        # Warning and optimization for large number of time steps
        if T > 5000:
            print(
                f"Warning: Too many time steps ({T}). Applying memory efficiency optimization."
            )
            if not hasattr(self, "use_memory_optimization"):
                self.use_memory_optimization = True
            print(
                f"Due to very many time steps ({T}), forcing sampling-based processing."
            )
            self.force_sampling = True
        else:
            self.use_memory_optimization = False
            self.force_sampling = False

        self.beta_values = [1.0]
        self.current_beta = self.beta
        self.temporal_graph = TemporalGraph(swap_cost=self.current_beta)

        # 並列化されたスナップショット構築
        self._build_snapshots_parallel(edges, node_list, node_map, T, n)

        self.node_num = n
        self.time_steps = T
        self.node_map = node_map
        self.orig_nodes = node_list

    def _build_snapshots_parallel(self, edges, node_list, node_map, T, n):
        """Parallelized snapshot construction (memory efficient version)"""
        print(f"Starting parallelized snapshot construction (n_jobs={self.n_jobs})")

        # Adjust chunk size for memory efficiency
        if self.use_memory_optimization:
            chunk_size = max(1, min(50, T // self.n_jobs))
        else:
            chunk_size = max(1, T // self.n_jobs)

        print(f"Chunk size: {chunk_size}")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Parallelize snapshot construction for each time step
            snapshot_futures = []
            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)
                future = executor.submit(
                    self._build_snapshots_chunk, edges, node_map, t_start, t_end, n
                )
                snapshot_futures.append(future)

            # Collect results
            for t_start, future in enumerate(snapshot_futures):
                snapshots = future.result()
                t_actual_start = t_start * chunk_size
                for i, A in enumerate(snapshots):
                    t = t_actual_start + i
                    if t < T:
                        self.temporal_graph.add_snapshot(A, node_list)

        print(f"Snapshot construction complete: {T} time steps")

    def _build_snapshots_chunk(self, edges, node_map, t_start, t_end, n):
        """Snapshot construction in chunk units"""
        snapshots = []

        for t in range(t_start, t_end):
            rows, cols, vals = [], [], []
            for s, d, time in edges:
                if time == t:
                    i, j = node_map[s], node_map[d]
                    rows += [i, j]
                    cols += [j, i]
                    vals += [1.0, 1.0]

            if rows:
                A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
                A.data = np.ones_like(A.data)
            else:
                A = sp.csr_matrix((n, n))

            snapshots.append(A)

        return snapshots

    def _analyze_data_characteristics(
        self, edges=None, node_list=None, node_map=None, time_map=None
    ) -> Dict:
        """
        Analyze data characteristics and compute metrics for beta adjustment

        Args:
            edges: Edge list (reloaded if None)
            node_list: Node list
            node_map: Node map
            time_map: Time map

        Returns:
            Dictionary of data characteristics
        """
        # Reload if arguments are not provided
        if edges is None:
            edges, node_list, node_map, time_map = load_edge_list_simple(self.edge_path)

        n = len(node_list)
        T = len(time_map)

        # Calculate edge density for each time step
        edge_densities = []
        for t in range(T):
            edges_at_t = [e for e in edges if e[2] == t]
            if n > 1:
                density = len(edges_at_t) / (n * (n - 1) / 2)
            else:
                density = 0.0
            edge_densities.append(density)

        # Calculate temporal change rate
        temporal_changes = []
        for t in range(T - 1):
            edges_t = set((e[0], e[1]) for e in edges if e[2] == t)
            edges_t1 = set((e[0], e[1]) for e in edges if e[2] == t + 1)

            # Measure change using Jaccard distance
            union = len(edges_t | edges_t1)
            if union > 0:
                change_rate = 1 - len(edges_t & edges_t1) / union
            else:
                change_rate = 0.0
            temporal_changes.append(change_rate)

        # Average edge density and temporal change rate
        avg_density = np.mean(edge_densities)
        avg_temporal_change = np.mean(temporal_changes) if temporal_changes else 0.0

        return {
            "avg_edge_density": avg_density,
            "avg_temporal_change": avg_temporal_change,
            "edge_densities": edge_densities,
            "temporal_changes": temporal_changes,
            "num_nodes": n,
            "num_timesteps": T,
            "total_edges": len(edges),
        }

    def _suggest_beta(self, characteristics: Dict) -> float:
        """
        Suggest appropriate beta value based on data characteristics

        Args:
            characteristics: Dictionary of data characteristics

        Returns:
            Suggested beta value
        """
        avg_density = characteristics["avg_edge_density"]
        avg_temporal_change = characteristics["avg_temporal_change"]
        n = characteristics["num_nodes"]
        T = characteristics["num_timesteps"]

        # Higher edge density leads to higher swap cost
        # Higher temporal change leads to lower swap cost
        density_factor = 1.0 + 2.0 * avg_density
        temporal_factor = 1.0 - 0.5 * avg_temporal_change

        # Also consider graph size and number of time steps
        size_factor = np.sqrt(n) / 10.0  # Higher with more nodes
        time_factor = 1.0 / np.sqrt(T)  # Lower with more time steps

        suggested_beta = density_factor * temporal_factor * size_factor * time_factor

        # Limit to reasonable range
        suggested_beta = max(0.1, min(10.0, suggested_beta))

        return suggested_beta

    def _power_method(
        self,
        matrix: sp.spmatrix,
        init_vector: np.ndarray = None,
        find_largest: bool = True,
    ) -> Tuple[float, np.ndarray]:
        n = matrix.shape[0]
        v = init_vector.copy() if init_vector is not None else np.random.randn(n)
        v /= np.linalg.norm(v)
        for _ in range(self.max_iter):
            v_prev = v.copy()
            if find_largest:
                v = matrix @ v
            else:
                try:
                    v = spla.spsolve(matrix, v)
                except:
                    v = spla.spsolve(matrix + 1e-6 * sp.identity(n), v)
            v /= np.linalg.norm(v)
            if np.linalg.norm(v - v_prev) < self.tolerance:
                break
        eigenval = float(v.T @ (matrix @ v))
        return eigenval, v

    def _sweep_cut_sparse(self, eigenvector: np.ndarray) -> Dict:
        """
        Sweep cut for sparse cut (optimized version)

        Sort by eigenvector values and find optimal cut
        Objective function: (edge cut + swap) / (|S| * |V-S|)

        Args:
            eigenvector: Eigenvector (nT dimensions)

        Returns:
            Dictionary of cut information (cut, score, edges, swaps)
        """
        nT = len(eigenvector)
        n = self.temporal_graph.size()
        T = self.temporal_graph.num_snapshots()

        # Use sampling-based acceleration for many time steps
        if T > 1000 and not self.force_exact_computation:
            return self._sampling_sweep_cut_sparse(eigenvector, nT, n, T)
        else:
            return self._parallel_sweep_cut_sparse(eigenvector, nT, n, T)

    def _sampling_sweep_cut_sparse(
        self, eigenvector: np.ndarray, nT: int, n: int, T: int
    ) -> Dict:
        """Sampling-based fast sweep cut (for large-scale data with progress display)"""
        print("  Executing sampling-based sweep cut...")

        # Sort by eigenvector values
        sorted_indices = np.argsort(eigenvector)

        # Calculate sampling interval (for memory efficiency)
        sample_interval = max(1, nT // 10000)  # Maximum 10000 samples
        sample_indices = sorted_indices[::sample_interval]

        print(
            f"    Sampling interval: {sample_interval}, sample count: {len(sample_indices)}"
        )

        # Parallelized sampling evaluation
        return self._parallel_sampling_evaluation_sparse(
            eigenvector, sample_indices, n, T, nT
        )

    def _parallel_sampling_evaluation_sparse(
        self,
        eigenvector: np.ndarray,
        sample_indices: np.ndarray,
        n: int,
        T: int,
        nT: int,
    ) -> Dict:
        """Parallelized sampling evaluation"""
        chunk_size = max(1, len(sample_indices) // self.n_jobs)

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for i in range(0, len(sample_indices), chunk_size):
                end_idx = min(i + chunk_size, len(sample_indices))
                future = executor.submit(
                    self._evaluate_sampling_range_sparse,
                    eigenvector,
                    sample_indices[i:end_idx],
                    n,
                    T,
                    nT,
                )
                futures.append(future)

            # Collect results
            cut_results = [future.result() for future in futures]

        # Select optimal cut
        return self._select_best_cut_sparse(cut_results, nT, n, T)

    def _evaluate_sampling_range_sparse(
        self,
        eigenvector: np.ndarray,
        sample_indices: np.ndarray,
        n: int,
        T: int,
        nT: int,
    ) -> Dict:
        """Evaluation of sampling range"""
        sizes_in_cut = np.zeros(T, dtype=int)
        nodes_in_cut = [set() for _ in range(T)]
        edges_cut = 0
        swaps = 0
        best_score = float("inf")
        best_cut = None
        best_edges_cut = 0
        best_swaps = 0

        for idx in sample_indices:
            t = idx // n
            v = idx % n
            sizes_in_cut[t] += 1
            nodes_in_cut[t].add(v)

            # Update edge cut (approximate computation)
            G_t = self.temporal_graph.get_snapshot(t)
            for neighbor in G_t.neighbors(v):
                weight = G_t[v][neighbor].get("weight", 1.0)
                if neighbor not in nodes_in_cut[t]:
                    edges_cut += weight
                else:
                    edges_cut -= weight

            # Update swap cost
            if t > 0:
                if v not in nodes_in_cut[t - 1]:
                    swaps += self.temporal_graph.swap_cost
                else:
                    swaps -= self.temporal_graph.swap_cost
            if t < T - 1:
                if v not in nodes_in_cut[t + 1]:
                    swaps += self.temporal_graph.swap_cost
                else:
                    swaps -= self.temporal_graph.swap_cost

            # Score calculation
            denominator = 0
            for tau in range(T):
                denominator += sizes_in_cut[tau] * (n - sizes_in_cut[tau])

            if denominator > 0:
                score = (edges_cut + swaps) / denominator
                if score < best_score:
                    best_score = score
                    best_cut = np.zeros(nT)
                    for tau in range(T):
                        for u in nodes_in_cut[tau]:
                            best_cut[tau * n + u] = -1.0
                        for u in range(n):
                            if u not in nodes_in_cut[tau]:
                                best_cut[tau * n + u] = 1.0
                    best_edges_cut = edges_cut
                    best_swaps = swaps

        return {
            "cut": best_cut,
            "score": best_score,
            "edges": best_edges_cut,
            "swaps": best_swaps,
        }

    def _select_best_cut_sparse(
        self, cut_results: List[Dict], nT: int, n: int, T: int
    ) -> Dict:
        """Select optimal cut"""
        best_score = float("inf")
        best_result = None

        for result in cut_results:
            if result["score"] < best_score:
                best_score = result["score"]
                best_result = result

        if best_result is None:
            return {
                "cut": np.ones(nT),
                "score": 0.0,
                "edges": 0,
                "swaps": 0,
            }

        return best_result

    def _sweep_cut_normalized(self, eigenvector: np.ndarray) -> Dict:
        """
        Sweep cut for normalized cut (optimized version)

        Objective function: (edge cut + swap) / (vol(S) * vol(V-S))
        where vol(S) is the volume of set S (sum of degrees)

        Args:
            eigenvector: Eigenvector (nT dimensions)

        Returns:
            Dictionary of cut information
        """
        nT = len(eigenvector)
        n = self.temporal_graph.size()
        T = self.temporal_graph.num_snapshots()

        # Use sampling-based acceleration for many time steps
        if T > 1000 and not self.force_exact_computation:
            return self._sampling_sweep_cut_normalized(eigenvector, nT, n, T)
        else:
            return self._parallel_sweep_cut_normalized(eigenvector, nT, n, T)

    def _sampling_sweep_cut_normalized(
        self, eigenvector: np.ndarray, nT: int, n: int, T: int
    ) -> Dict:
        """Sampling-based fast normalized sweep cut (with progress display)"""
        print("  Executing sampling-based normalized sweep cut...")

        # Sort by eigenvector values
        sorted_indices = np.argsort(eigenvector)

        # Calculate sampling interval
        sample_interval = max(1, nT // 10000)
        sample_indices = sorted_indices[::sample_interval]

        print(
            f"    Sampling interval: {sample_interval}, sample count: {len(sample_indices)}"
        )

        # Pre-calculate volume for each time step
        total_volumes = np.zeros(T)
        node_degrees = np.zeros((T, n))

        print("    Calculating volumes...")
        for t in range(T):
            G_t = self.temporal_graph.get_snapshot(t)
            for v in range(n):
                if v in G_t:
                    degree = sum(G_t[v][u].get("weight", 1.0) for u in G_t.neighbors(v))
                    node_degrees[t, v] = degree
                    total_volumes[t] += degree

        print("    Volume calculation complete")

        # Parallelized sampling evaluation
        return self._parallel_sampling_evaluation_normalized(
            eigenvector, sample_indices, n, T, node_degrees, total_volumes, nT
        )

    def _parallel_sampling_evaluation_normalized(
        self,
        eigenvector: np.ndarray,
        sample_indices: np.ndarray,
        n: int,
        T: int,
        node_degrees: np.ndarray,
        total_volumes: np.ndarray,
        nT: int,
    ) -> Dict:
        """Parallelized normalized sampling evaluation"""
        chunk_size = max(1, len(sample_indices) // self.n_jobs)

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for i in range(0, len(sample_indices), chunk_size):
                end_idx = min(i + chunk_size, len(sample_indices))
                future = executor.submit(
                    self._evaluate_sampling_range_normalized,
                    eigenvector,
                    sample_indices[i:end_idx],
                    n,
                    T,
                    node_degrees,
                    total_volumes,
                    nT,
                )
                futures.append(future)

            # Collect results
            cut_results = [future.result() for future in futures]

        # Select optimal cut
        return self._select_best_cut_normalized(cut_results, nT, n, T)

    def _parallel_sweep_cut_sparse(
        self, eigenvector: np.ndarray, nT: int, n: int, T: int
    ) -> Dict:
        """Parallelized sparse sweep cut (for small-scale data)"""
        # Sort by eigenvector values
        sorted_indices = np.argsort(eigenvector)

        # Divide into chunks for parallel processing
        chunk_size = max(1, nT // self.n_jobs)
        cut_points = list(range(0, nT, chunk_size))

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            cut_futures = []
            for i in range(len(cut_points) - 1):
                start_idx = cut_points[i]
                end_idx = cut_points[i + 1]
                future = executor.submit(
                    self._evaluate_cut_range_sparse,
                    eigenvector,
                    sorted_indices,
                    start_idx,
                    end_idx,
                    n,
                    T,
                    nT,
                )
                cut_futures.append(future)

            # Collect results
            cut_results = [future.result() for future in cut_futures]

        # Select optimal cut
        return self._select_best_cut_sparse(cut_results, nT, n, T)

    def _evaluate_cut_range_sparse(
        self,
        eigenvector: np.ndarray,
        sorted_indices: np.ndarray,
        start_idx: int,
        end_idx: int,
        n: int,
        T: int,
        nT: int,
    ) -> Dict:
        """Evaluation of cut range (for small-scale data)"""
        sizes_in_cut = np.zeros(T, dtype=int)
        nodes_in_cut = [set() for _ in range(T)]
        edges_cut = 0
        swaps = 0
        best_score = float("inf")
        best_cut = None
        best_edges_cut = 0
        best_swaps = 0
        for i in range(start_idx, end_idx):
            idx = sorted_indices[i]
            t = idx // n
            v = idx % n
            sizes_in_cut[t] += 1
            nodes_in_cut[t].add(v)
            G_t = self.temporal_graph.get_snapshot(t)
            for neighbor in G_t.neighbors(v):
                weight = G_t[v][neighbor].get("weight", 1.0)
                if neighbor not in nodes_in_cut[t]:
                    edges_cut += weight
                else:
                    edges_cut -= weight
            if t > 0:
                if v not in nodes_in_cut[t - 1]:
                    swaps += self.temporal_graph.swap_cost
                else:
                    swaps -= self.temporal_graph.swap_cost
            if t < T - 1:
                if v not in nodes_in_cut[t + 1]:
                    swaps += self.temporal_graph.swap_cost
                else:
                    swaps -= self.temporal_graph.swap_cost
            denominator = 0
            for tau in range(T):
                denominator += sizes_in_cut[tau] * (n - sizes_in_cut[tau])
            if denominator > 0:
                score = (edges_cut + swaps) / denominator
                if score < best_score:
                    best_score = score
                    best_cut = np.zeros(nT)
                    for tau in range(T):
                        for u in nodes_in_cut[tau]:
                            best_cut[tau * n + u] = -1.0
                        for u in range(n):
                            if u not in nodes_in_cut[tau]:
                                best_cut[tau * n + u] = 1.0
                    best_edges_cut = edges_cut
                    best_swaps = swaps
        return {
            "cut": best_cut,
            "score": best_score,
            "edges": best_edges_cut,
            "swaps": best_swaps,
        }

    def _parallel_sweep_cut_normalized(
        self, eigenvector: np.ndarray, nT: int, n: int, T: int
    ) -> Dict:
        """Parallelized normalized sweep cut (for small-scale data)"""
        sorted_indices = np.argsort(eigenvector)

        # Pre-calculate volume for each time step
        total_volumes = np.zeros(T)
        node_degrees = np.zeros((T, n))

        for t in range(T):
            G_t = self.temporal_graph.get_snapshot(t)
            for v in range(n):
                if v in G_t:
                    degree = sum(G_t[v][u].get("weight", 1.0) for u in G_t.neighbors(v))
                    node_degrees[t, v] = degree
                    total_volumes[t] += degree

        # Divide into chunks for parallel processing
        chunk_size = max(1, nT // self.n_jobs)
        cut_points = list(range(0, nT, chunk_size))

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            cut_futures = []
            for i in range(len(cut_points) - 1):
                start_idx = cut_points[i]
                end_idx = cut_points[i + 1]
                future = executor.submit(
                    self._evaluate_cut_range_normalized,
                    eigenvector,
                    sorted_indices,
                    start_idx,
                    end_idx,
                    n,
                    T,
                    node_degrees,
                    total_volumes,
                    nT,
                )
                cut_futures.append(future)

            # Collect results
            cut_results = [future.result() for future in cut_futures]

        # Select optimal cut
        return self._select_best_cut_normalized(cut_results, nT, n, T)

    def _evaluate_cut_range_normalized(
        self,
        eigenvector: np.ndarray,
        sorted_indices: np.ndarray,
        start_idx: int,
        end_idx: int,
        n: int,
        T: int,
        node_degrees: np.ndarray,
        total_volumes: np.ndarray,
        nT: int,
    ) -> Dict:
        """Evaluation of normalized cut range (for small-scale data)"""
        volumes_in_cut = np.zeros(T)
        nodes_in_cut = [set() for _ in range(T)]
        edges_cut = 0
        swaps = 0
        best_score = float("inf")
        best_cut = None
        best_edges_cut = 0
        best_swaps = 0
        for i in range(start_idx, end_idx):
            idx = sorted_indices[i]
            t = idx // n
            v = idx % n
            volumes_in_cut[t] += node_degrees[t, v]
            nodes_in_cut[t].add(v)
            G_t = self.temporal_graph.get_snapshot(t)
            for neighbor in G_t.neighbors(v):
                weight = G_t[v][neighbor].get("weight", 1.0)
                if neighbor not in nodes_in_cut[t]:
                    edges_cut += weight
                else:
                    edges_cut -= weight
            if t > 0:
                if v not in nodes_in_cut[t - 1]:
                    swaps += self.temporal_graph.swap_cost
                else:
                    swaps -= self.temporal_graph.swap_cost
            if t < T - 1:
                if v not in nodes_in_cut[t + 1]:
                    swaps += self.temporal_graph.swap_cost
                else:
                    swaps -= self.temporal_graph.swap_cost
            denominator = 0
            for tau in range(T):
                vol_in = volumes_in_cut[tau]
                vol_out = total_volumes[tau] - vol_in
                denominator += vol_in * vol_out
            if denominator > 0:
                score = (edges_cut + swaps) / denominator
                if score < best_score:
                    best_score = score
                    best_cut = np.zeros(nT)
                    for tau in range(T):
                        for u in nodes_in_cut[tau]:
                            best_cut[tau * n + u] = -1.0
                        for u in range(n):
                            if u not in nodes_in_cut[tau]:
                                best_cut[tau * n + u] = 1.0
                    best_edges_cut = edges_cut
                    best_swaps = swaps
        return {
            "cut": best_cut,
            "score": best_score,
            "edges": best_edges_cut,
            "swaps": best_swaps,
        }

    def _select_best_cut_normalized(
        self, cut_results: List[Dict], nT: int, n: int, T: int
    ) -> Dict:
        """Select optimal normalized cut"""
        best_score = float("inf")
        best_result = None

        for result in cut_results:
            if result["score"] < best_score:
                best_score = result["score"]
                best_result = result

        if best_result is None:
            return {
                "cut": np.ones(nT),
                "score": 0.0,
                "edges": 0,
                "swaps": 0,
            }

        return best_result

    def _diff_cut_method(self) -> Tuple[np.ndarray, Dict]:
        L = self.temporal_graph.create_laplacian_matrix(self.cut_type == "normalized")
        C = self.temporal_graph.create_complete_laplacian_matrix()
        n = self.temporal_graph.size()
        coeff = 3.0 * (n + 2.0 * self.beta)
        M = coeff * C - L
        try:
            vals, vecs = spla.eigsh(M, k=1, which="LA")
            v = vecs[:, 0].real
            eig = vals[0].real
        except:
            eig, v = self._power_method(M, find_largest=True)
        cut = (
            self._sweep_cut_sparse(v)
            if self.cut_type == "sparse"
            else self._sweep_cut_normalized(v)
        )
        self._eigenvalues = np.array([eig])
        self._eigenvectors = v.reshape(-1, 1)
        return v, cut

    def _laplacian_cut_method(self) -> Tuple[np.ndarray, Dict]:
        L = self.temporal_graph.create_laplacian_matrix(self.cut_type == "normalized")
        try:
            vals, vecs = spla.eigsh(L, k=2, which="SA")
            v = vecs[:, 1].real
            eig = vals[1].real
        except:
            eig, v = self._power_method(
                L + 1e-3 * sp.identity(L.shape[0]), find_largest=False
            )
        cut = (
            self._sweep_cut_sparse(v)
            if self.cut_type == "sparse"
            else self._sweep_cut_normalized(v)
        )
        self._eigenvalues = np.array([eig])
        self._eigenvectors = v.reshape(-1, 1)
        return v, cut

    def _prod_cut_method(self) -> Tuple[np.ndarray, Dict]:
        L = self.temporal_graph.create_laplacian_matrix(self.cut_type == "normalized")
        C = self.temporal_graph.create_complete_laplacian_matrix()
        try:
            M = (C @ L) @ C.T
        except MemoryError:
            return self._diff_cut_method()
        try:
            vals, vecs = spla.eigsh(M, k=1, which="LA")
            v = vecs[:, 0].real
            eig = vals[0].real
        except:
            eig, v = self._power_method(M, find_largest=True)
        cut = (
            self._sweep_cut_sparse(v)
            if self.cut_type == "sparse"
            else self._sweep_cut_normalized(v)
        )
        self._eigenvalues = np.array([eig])
        self._eigenvectors = v.reshape(-1, 1)
        return v, cut

    def _fast_cut_method(self) -> Tuple[np.ndarray, Dict]:
        T, n, r = self.time_steps, self.node_num, min(self.rank_r, self.node_num - 1)

        print(f"Fast cut method: T={T}, n={n}, r={r}")
        print("Starting parallelized Laplacian computation...")

        # Parallelized Laplacian computation
        vals_list, vecs_list = self._compute_laplacians_fast_parallel(r)

        print("Constructing low-rank approximation matrix...")
        print("  Step 1/3: Coefficient calculation")
        c = 3.0 * (n + 2.0 * self.beta) * n
        print(f"    Coefficient c = {c:.2e}")

        # Parallelized block diagonal matrix construction
        print("  Step 2/3: Block diagonal matrix construction")
        Λ = self._build_block_diagonal_parallel(vals_list, c, T, r)

        # Construction of temporal connection matrix B
        print("  Step 3/3: Temporal connection matrix construction")
        B = self._build_temporal_connections_fast(T, r)

        print("Low-rank approximation matrix construction complete")
        print(f"Solving low-rank eigenvalue problem... (matrix size: {T*r}×{T*r})")
        M_small = Λ - B

        try:
            vals, vecs = spla.eigsh(M_small, k=1, which="LA")
            small = vecs[:, 0].real
            eig = vals[0].real
        except:
            eig, small = self._power_method(M_small, find_largest=True)

        print("Reconstructing full-dimensional vector...")
        # Parallelized full-dimensional vector reconstruction
        full = self._reconstruct_full_vector_parallel(vecs_list, small, T, n, r)

        print("Computing sweep cut...")
        cut = (
            self._sweep_cut_sparse(full)
            if self.cut_type == "sparse"
            else self._sweep_cut_normalized(full)
        )

        self._eigenvalues = np.array([eig])
        self._eigenvectors = full.reshape(-1, 1)
        return full, cut

    def _compute_laplacians_fast_parallel(
        self, r: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Parallelized Laplacian computation (for fast method with progress display)"""
        T = self.time_steps
        vals_list, vecs_list = [], []

        # Adjust chunk size (for memory efficiency)
        chunk_size = max(1, min(100, T // self.n_jobs))
        total_chunks = (T + chunk_size - 1) // chunk_size

        print(f"  Laplacian computation: {T} time steps, {total_chunks} chunks")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for t in range(0, T, chunk_size):
                end_t = min(t + chunk_size, T)
                future = executor.submit(self._compute_laplacians_chunk, t, end_t, r)
                futures.append(future)

            # Collect results (with progress display)
            completed_chunks = 0
            for future in futures:
                chunk_vals, chunk_vecs = future.result()
                vals_list.extend(chunk_vals)
                vecs_list.extend(chunk_vecs)
                completed_chunks += 1

                # Progress display (every 10%)
                if (
                    completed_chunks % max(1, total_chunks // 10) == 0
                    or completed_chunks == total_chunks
                ):
                    progress = (completed_chunks / total_chunks) * 100
                    print(
                        f"    Progress: {completed_chunks}/{total_chunks} chunks complete ({progress:.1f}%)"
                    )

        print(f"  Laplacian computation complete: {len(vals_list)} time steps")
        return vals_list, vecs_list

    def _compute_laplacians_chunk(
        self, start_t: int, end_t: int, r: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Laplacian computation in chunk units"""
        vals_list, vecs_list = [], []

        for t in range(start_t, end_t):
            A = nx.adjacency_matrix(
                self.temporal_graph.get_snapshot(t),
                nodelist=self.orig_nodes,
                weight="weight",
            )
            L_t = (
                n1_laplacian(A)
                if self.cut_type == "normalized"
                else unnormalized_laplacian(A)
            )
            try:
                vals, vecs = spla.eigsh(L_t, k=r, which="SA")
            except:
                # Calculate minimum necessary eigenvalues for memory efficiency
                if self.force_exact_computation or L_t.shape[0] <= 1000:
                    vals_full, vecs_full = la.eigh(L_t.toarray())
                    vals, vecs = vals_full[:r], vecs_full[:, :r]
                else:
                    # Use approximate computation for large-scale graphs
                    vals, vecs = self._approximate_eigenvalues(L_t, r)

            vals_list.append(vals)
            vecs_list.append(vecs)

        return vals_list, vecs_list

    def _approximate_eigenvalues(
        self, L_t: sp.spmatrix, r: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate eigenvalue computation for large-scale graphs"""
        n = L_t.shape[0]

        # Random initialization
        Q = np.random.randn(n, min(2 * r, n))
        Q, _ = np.linalg.qr(Q)

        # Approximation using power method
        for _ in range(10):
            Q_new = L_t @ Q
            Q, R = np.linalg.qr(Q_new)

        # Eigenvalue decomposition of projected matrix
        H = Q.T @ L_t @ Q
        vals, vecs_small = np.linalg.eigh(H)

        # Get top r eigenvalues and eigenvectors
        idx = np.argsort(vals)[:r]
        vals = vals[idx]
        vecs = Q @ vecs_small[:, idx]

        return vals, vecs

    def _build_block_diagonal_parallel(
        self, vals_list: List[np.ndarray], c: float, T: int, r: int
    ) -> sp.spmatrix:
        """Parallelized block diagonal matrix construction (with progress display)"""
        blocks = []
        total_blocks = len(vals_list)

        print(f"  Block diagonal matrix construction start: {total_blocks} blocks")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for t, vals in enumerate(vals_list):
                future = executor.submit(self._build_single_block, vals, c, r)
                futures.append(future)

            # Collect results (with progress display)
            completed_blocks = 0
            for i, future in enumerate(futures):
                block = future.result()
                blocks.append(block)
                completed_blocks += 1

                # Progress display (every 10%)
                if (
                    completed_blocks % max(1, total_blocks // 10) == 0
                    or completed_blocks == total_blocks
                ):
                    progress = (completed_blocks / total_blocks) * 100
                    print(
                        f"    Progress: {completed_blocks}/{total_blocks} blocks complete ({progress:.1f}%)"
                    )

        print("  Block diagonal matrix construction complete")
        return sp.block_diag(blocks, format="csr")

    def _build_single_block(self, vals: np.ndarray, c: float, r: int) -> sp.spmatrix:
        """Construction of single block"""
        shifted = c - vals
        shifted[0] = 0.0
        return sp.diags(shifted)

    def _build_temporal_connections_fast(self, T: int, r: int) -> sp.spmatrix:
        """Fast construction of temporal connection matrix"""
        # Pre-calculate required number of elements
        total_elements = 4 * (T - 1) * r

        if total_elements > 1e8:  # When memory usage is large
            return self._build_temporal_connections_sparse(T, r)
        else:
            return self._build_temporal_connections_dense(T, r)

    def _build_temporal_connections_sparse(self, T: int, r: int) -> sp.spmatrix:
        """Sparse temporal connection matrix construction (memory efficient version with progress display)"""
        B_rows, B_cols, B_vals = [], [], []

        # Process in chunks
        chunk_size = max(1, min(1000, (T - 1) // self.n_jobs))
        total_chunks = (T - 1 + chunk_size - 1) // chunk_size

        print(f"  Temporal connection matrix construction start: {T-1} time intervals, {total_chunks} chunks")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for t_start in range(0, T - 1, chunk_size):
                t_end = min(t_start + chunk_size, T - 1)
                future = executor.submit(self._build_temporal_chunk, t_start, t_end, r)
                futures.append(future)

            # Collect results (with progress display)
            completed_chunks = 0
            for future in futures:
                chunk_rows, chunk_cols, chunk_vals = future.result()
                B_rows.extend(chunk_rows)
                B_cols.extend(chunk_cols)
                B_vals.extend(chunk_vals)
                completed_chunks += 1

                # Progress display (every 10%)
                if (
                    completed_chunks % max(1, total_chunks // 10) == 0
                    or completed_chunks == total_chunks
                ):
                    progress = (completed_chunks / total_chunks) * 100
                    print(
                        f"    Progress: {completed_chunks}/{total_chunks} chunks complete ({progress:.1f}%)"
                    )

        print(f"  Temporal connection matrix construction complete: {len(B_vals)} elements")
        return sp.coo_matrix((B_vals, (B_rows, B_cols)), shape=(T * r, T * r)).tocsr()

    def _build_temporal_chunk(
        self, t_start: int, t_end: int, r: int
    ) -> Tuple[List[int], List[int], List[float]]:
        """Temporal connection construction in chunk units"""
        rows, cols, vals = [], [], []

        for t in range(t_start, t_end):
            for i in range(r):
                i1, i2 = t * r + i, (t + 1) * r + i
                rows += [i1, i2, i1, i2]
                cols += [i2, i1, i1, i2]
                vals += [self.beta, self.beta, self.beta, self.beta]

        return rows, cols, vals

    def _build_temporal_connections_dense(self, T: int, r: int) -> sp.spmatrix:
        """Dense temporal connection matrix construction (for small-scale)"""
        B_rows, B_cols, B_vals = [], [], []
        for t in range(T - 1):
            for i in range(r):
                i1, i2 = t * r + i, (t + 1) * r + i
                B_rows += [i1, i2, i1, i2]
                B_cols += [i2, i1, i1, i2]
                B_vals += [self.beta, self.beta, self.beta, self.beta]
        return sp.coo_matrix((B_vals, (B_rows, B_cols)), shape=(T * r, T * r)).tocsr()

    def _reconstruct_full_vector_parallel(
        self, vecs_list: List[np.ndarray], small: np.ndarray, T: int, n: int, r: int
    ) -> np.ndarray:
        """Parallelized full-dimensional vector reconstruction (with progress display)"""
        full = np.zeros(n * T)

        # Parallel processing in chunks
        chunk_size = max(1, min(100, T // self.n_jobs))
        total_chunks = (T + chunk_size - 1) // chunk_size

        print(f"  Vector reconstruction: {T} time steps, {total_chunks} chunks")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)
                future = executor.submit(
                    self._reconstruct_vector_chunk,
                    vecs_list,
                    small,
                    t_start,
                    t_end,
                    n,
                    r,
                )
                futures.append(future)

            # Collect results (with progress display)
            completed_chunks = 0
            for t_start, future in enumerate(futures):
                chunk_result = future.result()
                t_actual_start = t_start * chunk_size
                t_actual_end = min(t_actual_start + chunk_size, T)
                for t in range(t_actual_start, t_actual_end):
                    full[t * n:(t + 1) * n] = chunk_result[t - t_actual_start]

                completed_chunks += 1

                # Progress display (every 10%)
                if (
                    completed_chunks % max(1, total_chunks // 10) == 0
                    or completed_chunks == total_chunks
                ):
                    progress = (completed_chunks / total_chunks) * 100
                    print(
                        f"    Progress: {completed_chunks}/{total_chunks} chunks complete ({progress:.1f}%)"
                    )

        print("  Vector reconstruction complete")
        return full

    def _reconstruct_vector_chunk(
        self,
        vecs_list: List[np.ndarray],
        small: np.ndarray,
        t_start: int,
        t_end: int,
        n: int,
        r: int,
    ) -> List[np.ndarray]:
        """Vector reconstruction in chunk units"""
        chunk_vectors = []

        for t in range(t_start, t_end):
            U = vecs_list[t]
            vt = small[t * r:(t + 1) * r]
            chunk_vectors.append(U @ vt)

        return chunk_vectors

    def compute_embedding(self):
        print("\n=== TemporalCut embedding computation start ===")
        print(f"Method: {self.method}, cut type: {self.cut_type}")
        print(f"Embedding dimensions: {self.emb_size}")
        print(
            f"Parallelization settings: {self.n_jobs} parallel, sparse: {self.use_sparse}"
        )

        if hasattr(self, "use_memory_optimization") and self.use_memory_optimization:
            print("Memory optimization: enabled")
        if hasattr(self, "force_sampling") and self.force_sampling:
            print("Forced sampling: enabled")

        # Recommend fast method for many time steps
        if self.time_steps > 5000 and self.method != "fast":
            print(
                f"Warning: Due to many time steps ({self.time_steps}), fast method is recommended."
            )
            if self.method in ["diff", "laplacian", "prod"]:
                print(f"Current method {self.method} may be slow for large-scale data.")

        # Experiment with multiple beta values
        self.all_embeddings = {}
        self.all_cut_infos = {}

        print("\n=== Multiple beta value experiment start ===")
        print(f"Beta values: {self.beta_values}")
        print(f"Loop start: processing {len(self.beta_values)} beta values")

        for i, beta in enumerate(self.beta_values):
            print(
                f"\n--- Computation start for beta = {beta} ({i+1}/{len(self.beta_values)}) ---"
            )
            self.current_beta = beta
            self.temporal_graph.swap_cost = beta
            print(f"  swap_cost set to {beta} complete")

            # Embedding computation for each beta value
            try:
                print(f"  Embedding computation start: method = {self.method}")
                if self.method == "diff":
                    base_vec, cut_info = self._diff_cut_method()
                elif self.method == "laplacian":
                    base_vec, cut_info = self._laplacian_cut_method()
                elif self.method == "prod":
                    base_vec, cut_info = self._prod_cut_method()
                elif self.method == "fast":
                    base_vec, cut_info = self._fast_cut_method()
                else:
                    raise ValueError(f"Unknown method: {self.method}")

                print(
                    "  Base vector computation complete, starting dimension calculation"
                )
                if self.emb_size > 1:
                    print(
                        f"  Parallel computation of multi-dimensional embedding... (remaining {self.emb_size-1} dimensions)"
                    )
                    embedding = self._compute_multiple_embeddings_parallel(base_vec)
                else:
                    embedding = base_vec.reshape(-1, 1)

                self.all_embeddings[beta] = embedding
                self.all_cut_infos[beta] = cut_info
                print(f"  Computation complete for beta = {beta}")
                print(
                    f"    Current keys in all_embeddings: {list(self.all_embeddings.keys())}"
                )

            except Exception as e:
                print(f"  Error occurred during computation for beta = {beta}: {e}")
                # Save empty result when error occurs
                self.all_embeddings[beta] = None
                self.all_cut_infos[beta] = {"error": str(e)}
                print(
                    f"    Current keys in all_embeddings: {list(self.all_embeddings.keys())}"
                )

        print(f"\n=== Loop end: All {len(self.beta_values)} beta values processed ===")

        # Summary of computation results
        successful_betas = [
            beta
            for beta in self.beta_values
            if beta in self.all_embeddings and self.all_embeddings[beta] is not None
        ]
        print("\n=== Computation complete for all beta values ===")
        print(f"Successful beta values: {successful_betas}")
        print(f"Keys in all_embeddings: {list(self.all_embeddings.keys())}")
        if len(successful_betas) < len(self.beta_values):
            failed_betas = [
                beta
                for beta in self.beta_values
                if beta not in self.all_embeddings or self.all_embeddings[beta] is None
            ]
            print(f"Failed beta values: {failed_betas}")
            print("Failure reasons:")
            for beta in failed_betas:
                if beta not in self.all_embeddings:
                    print(f"  beta = {beta}: key does not exist")
                elif self.all_embeddings[beta] is None:
                    print(f"  beta = {beta}: value is None (error occurred)")

        # Do not set default embedding (optimal beta value cannot be determined in advance)
        self.embedding = None

    def _compute_multiple_embeddings_parallel(self, base_vec: np.ndarray) -> np.ndarray:
        """Parallelized multi-dimensional embedding computation"""
        all_vecs = [base_vec]

        # Parallel computation of remaining dimensions
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            embedding_futures = []

            for i in range(1, self.emb_size):
                future = executor.submit(
                    self._compute_single_embedding_dimension,
                    i,
                    base_vec,
                    all_vecs,
                    self.method,
                )
                embedding_futures.append(future)

            # Collect results
            for i, future in enumerate(embedding_futures):
                next_vec = future.result()
                all_vecs.append(next_vec)
                print(f"  {i+2}th dimension computation complete")

        return np.column_stack(all_vecs)

    def _compute_single_embedding_dimension(
        self,
        dim_idx: int,
        base_vec: np.ndarray,
        prev_vecs: List[np.ndarray],
        method: str,
    ) -> np.ndarray:
        """Compute single embedding dimension (for parallelization)"""
        if method == "diff":
            L = self.temporal_graph.create_laplacian_matrix(
                self.cut_type == "normalized"
            )
            C = self.temporal_graph.create_complete_laplacian_matrix()
            n = self.temporal_graph.size()
            coeff = 3.0 * (n + 2.0 * self.beta)
            M = coeff * C - L
            try:
                vals, vecs = spla.eigsh(M, k=dim_idx + 1, which="LA")
                next_vec = vecs[:, dim_idx].real
            except Exception:
                init = np.random.randn(M.shape[0])
                for prev in prev_vecs:
                    init -= np.dot(init, prev) * prev
                init /= np.linalg.norm(init)
                _, next_vec = self._power_method(M, init, True)
        elif method == "laplacian":
            L = self.temporal_graph.create_laplacian_matrix(
                self.cut_type == "normalized"
            )
            try:
                vals, vecs = spla.eigsh(L, k=dim_idx + 2, which="SA")
                next_vec = vecs[:, dim_idx + 1].real
            except Exception:
                init = np.random.randn(L.shape[0])
                for prev in prev_vecs:
                    init -= np.dot(init, prev) * prev
                init /= np.linalg.norm(init)
                shifted = L + 1e-3 * sp.identity(L.shape[0])
                _, next_vec = self._power_method(shifted, init, False)
        else:
            next_vec = base_vec + 0.1 * np.random.randn(len(base_vec))

        return next_vec

    def save_node_embeddings(self, output_path=None):
        if output_path is None:
            output_path = self.output_path
        if not hasattr(self, "all_embeddings") or not self.all_embeddings:
            raise RuntimeError("Please call compute_embedding() first")

            # Save results for each beta value
        print("\n=== Embedding save start ===")
        print(f"Target beta values for saving: {self.beta_values}")
        print(f"Keys in all_embeddings: {list(self.all_embeddings.keys())}")

        saved_count = 0
        for beta in self.beta_values:
            # Skip if key does not exist
            if beta not in self.all_embeddings:
                print(f"Embedding for beta = {beta} will not be saved (not computed)")
                continue

            embedding = self.all_embeddings[beta]

            # Skip if error occurred
            if embedding is None:
                print(f"Embedding for beta = {beta} will not be saved (error occurred)")
                continue

            nT, d = embedding.shape
            n = self.node_num
            T = nT // n

            # Generate filename according to beta value
            base_path = output_path.rsplit(".", 1)[0]  # Remove extension
            beta_suffix = str(beta).replace(
                ".", "_"
            )  # Convert decimal point to underscore
            beta_output_path = f"{base_path}_{beta_suffix}.emb"

            print(f"Saving embedding for beta = {beta}: {beta_output_path}")
            os.makedirs(os.path.dirname(beta_output_path), exist_ok=True)

            with open(beta_output_path, "w") as f:
                f.write(f"{n} {d}\n")
                for t in range(T):
                    for v, orig in enumerate(self.orig_nodes):
                        idx = t * n + v
                        vals = " ".join(map(str, embedding[idx]))
                        f.write(f"{orig} {vals}\n")

            print(
                f"  beta = {beta} save complete: {n} nodes × {T} time steps × {d} dimensions"
            )
            saved_count += 1

        print(f"Embedding save complete for all beta values: {saved_count} files saved")

    def get_embedding_statistics(self) -> Dict:
        if not hasattr(self, "all_embeddings") or not self.all_embeddings:
            return {"status": "not_computed"}

        # Use statistics of successful beta values as baseline
        successful_betas = [
            beta
            for beta in self.beta_values
            if beta in self.all_embeddings and self.all_embeddings[beta] is not None
        ]
        if not successful_betas:
            return {"status": "all_failed"}

        # Use statistics of first successful beta value
        first_successful_beta = successful_betas[0]
        embedding = self.all_embeddings[first_successful_beta]
        stats = super().get_embedding_statistics()
        nT, d = embedding.shape
        n = self.node_num
        T = nT // n

        stats.update(
            {
                "cut_method": self.method,
                "cut_type": self.cut_type,
                "embedding_shape": (n, d, T),
                "swap_cost": first_successful_beta,
                "auto_adjust_beta": self.auto_adjust_beta,
                "node_count": n,
                "time_steps": T,
                "parallelization": {
                    "n_jobs": self.n_jobs,
                    "use_sparse": self.use_sparse,
                },
                "beta_experiment": {
                    "beta_values": self.beta_values,
                    "num_experiments": len(self.beta_values),
                    "all_cut_infos": self.all_cut_infos,
                    "successful_betas": successful_betas,
                },
            }
        )
        if self._eigenvalues is not None:
            stats["eigenvalues"] = self._eigenvalues.tolist()
        return stats
