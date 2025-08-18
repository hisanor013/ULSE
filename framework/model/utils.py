# type: ignore
import numpy as np
import scipy.sparse as sp
from typing import Union

Regularized = Union[str, float, bool]


def resolve_regularization_bias(regularized: Regularized, degrees: np.ndarray) -> float:
    if regularized in ("avg", True):
        return float(np.mean(degrees))
    elif isinstance(regularized, (float, int)):
        return float(regularized)
    elif regularized is False:
        return 0.0
    else:
        raise ValueError(f"Invalid regularized value: {regularized}")


def degree_vector(A, regularized=False):
    """
    Returns the degree vector of adjacency matrix A
    ・When regularized=True, returns the degree vector normalized as D+rI
    - (to prevent nodes with degree 0 from appearing)
    - r uses the average degree
    """
    if sp.issparse(A):
        # For sparse matrix, sum(axis=1) returns (n,1) matrix so flatten it
        degrees = np.array(A.sum(axis=1)).flatten()
    else:
        # For dense matrix
        degrees = A.sum(axis=1)

    degrees = degrees + resolve_regularization_bias(regularized, degrees)

    return degrees  # Return as 1D array


def aggregated_degree_vector(A_list, regularized=False):
    """
    Returns the aggregated degree vector of adjacency matrix A
    """
    if sp.issparse(A_list[0]):
        degrees = np.sum(np.array([A.sum(axis=1) for A in A_list]), axis=0)
    else:
        degrees = np.sum(np.array([A.sum(axis=1) for A in A_list]), axis=0)

    degrees = degrees + resolve_regularization_bias(regularized, degrees)

    return degrees


def n1_laplacian(A, regularized=False):
    """
    Returns the n1-normalized Laplacian (L = I - D^{-1/2} A D^{-1/2}) of adjacency matrix A.
    Supports both numpy.ndarray (dense) and scipy.sparse.spmatrix (sparse).
    """
    # Convert NetworkX sparse array to sparse matrix (for future compatibility)
    if hasattr(A, "toarray") and not sp.issparse(A):
        # For NetworkX 3.0 sparse array
        A = sp.csr_matrix(A)

    degrees = degree_vector(A, regularized)

    if isinstance(A, np.ndarray):  # dense
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:  # sparse
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees), format="csr")
        identity = sp.identity(A.shape[0], format="csr")
        A_csr = A.tocsr()  # Convert to CSR format
        return identity - D_inv_sqrt @ A_csr @ D_inv_sqrt


def n1_laplacian_list(A_list, regularized=False):
    """
    Returns a list of n1-normalized Laplacians (L = I - D^{-1/2} A D^{-1/2}) for adjacency matrices A corresponding to each time step.
    """
    return [n1_laplacian(A, regularized) for A in A_list]


def n2_laplacian(A, aggregated_degree_vector, regularized=False):
    """
    Returns the n2-normalized Laplacian (L = - D^{(1:T)}^{-1/2} A D^{-1/2}) of adjacency matrix A.
    Supports both numpy.ndarray (dense) and scipy.sparse.spmatrix (sparse).
    """
    # Convert NetworkX sparse array to sparse matrix (for future compatibility)
    if hasattr(A, "toarray") and not sp.issparse(A):
        # For NetworkX 3.0 sparse array
        A = sp.csr_matrix(A)

    aggregated_D_inv_sqrt = np.diag(1.0 / np.sqrt(aggregated_degree_vector))
    degrees = degree_vector(A, regularized)

    if isinstance(A, np.ndarray):  # dense
        D_t_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        return -aggregated_D_inv_sqrt @ A @ D_t_inv_sqrt
    else:  # sparse
        D_t_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees), format="csr")
        A_csr = A.tocsr()  # Convert to CSR format
        return -aggregated_D_inv_sqrt @ A_csr @ D_t_inv_sqrt


def n2_laplacian_list(A_list, regularized=False):
    """
    Returns a list of n2-normalized Laplacians (L = - D^{(1:T)}^{-1/2} A D^{-1/2}) for adjacency matrices A corresponding to each time step.
    """
    aggregated_degrees = aggregated_degree_vector(A_list, regularized)
    return [n2_laplacian(A, aggregated_degrees, regularized) for A in A_list]


# type: ignore
def unnormalized_laplacian(A, regularized=False):
    """
    Returns the unnormalized Laplacian (L = D - A) of adjacency matrix A.
    Supports both numpy.ndarray (dense) and scipy.sparse.spmatrix (sparse).
    """
    # Convert NetworkX sparse array to sparse matrix (for future compatibility)
    if hasattr(A, "toarray") and not sp.issparse(A):
        # For NetworkX 3.0 sparse array
        A = sp.csr_matrix(A)

    degrees = degree_vector(A, regularized)

    if isinstance(A, np.ndarray):  # dense
        D = np.diag(degrees)
        return D - A
    else:  # sparse
        D = sp.diags(degrees, format="csr")
        A_csr = A.tocsr()  # Convert to CSR format
        return D - A_csr


def tri_diagonal_time_laplacian(T: int) -> sp.spmatrix:
    """
    Return T×T 1-D chain Laplacian with homogeneous Neumann boundary
    ( 2  on diag except endpoints 1, and −1  on off-diag).
    """
    main = np.full(T, 2.0)
    if T >= 1:
        main[0] = main[-1] = 1.0
    off = np.full(T - 1, -1.0)
    return sp.diags([main, off, off], [0, -1, 1], format="csr")  # type: ignore


def preprocess_and_bin_edges(edge_path: str, out_path: str, time_bins: int = 100):
    """
    Loads edge list, performs time series binning, and saves the binned edge list to out_path.
    Save format: s t time_bin
    """
    import numpy as np

    edges = []
    node_set = set()
    time_set = set()
    with open(edge_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            s, t, time = line.strip().split()
            s, t, time = int(s), int(t), float(time)
            edges.append((s, t, time))
            node_set.update([s, t])
            time_set.add(time)
    node_list = sorted(list(node_set))
    time_list = sorted(list(time_set))
    if len(time_list) > time_bins:
        min_time, max_time = min(time_list), max(time_list)
        bins = np.linspace(min_time, max_time, time_bins + 1)
        new_edges = []
        for s, t, time in edges:
            new_time = np.digitize(time, bins) - 1
            if new_time >= time_bins:
                new_time = time_bins - 1
            new_edges.append((s, t, new_time))
        edges = new_edges
    else:
        # Use existing time as bin number directly
        time_map = {time: idx for idx, time in enumerate(time_list)}
        edges = [(s, t, time_map[time]) for s, t, time in edges]
    with open(out_path, "w") as f:
        for s, t, time in edges:
            f.write(f"{s} {t} {time}\n")


def load_edge_list_simple(edge_path: str):
    """
    Loads edge list (s t time format)
    Converts original time data to continuous integer indices

    Args:
        edge_path: Path to edge list file (s t time format)

    Returns:
        edges, node_list, node_map, time_map
    """
    edges = []
    node_set = set()
    time_set = set()
    with open(edge_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            s, t, time = line.strip().split()
            s, t, time = int(s), int(t), float(time)
            edges.append((s, t, time))
            node_set.update([s, t])
            time_set.add(time)

    node_list = sorted(list(node_set))
    node_map = {node: idx for idx, node in enumerate(node_list)}

    # Convert time to continuous integer indices
    time_list = sorted(list(time_set))
    time_map = {time: idx for idx, time in enumerate(time_list)}

    print("Time mapping information:")
    print(f"  Total number of time steps: {len(time_list)}")
    if len(time_list) <= 20:
        print(f"  Time mapping: {dict(list(time_map.items()))}")
    else:
        print(f"  Time range: {min(time_list):.2f} ～ {max(time_list):.2f}")
        print(f"  First 5: {dict(list(time_map.items())[:5])}")
        print(f"  Last 5: {dict(list(time_map.items())[-5:])}")

    # Convert edge times to indices
    processed_edges = []
    for s, t, time in edges:
        if time not in time_map:
            print(f"Error: Time {time} does not exist in time mapping")
            print(f"  Available times: {list(time_map.keys())}...")
            raise KeyError(f"Time {time} not found")
        processed_edges.append((s, t, time_map[time]))

    return processed_edges, node_list, node_map, time_map
