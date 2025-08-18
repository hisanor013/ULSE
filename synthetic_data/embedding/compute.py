from pathlib import Path
from typing import List, Dict
import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from framework.model.USE import USE

def compute_embeddings(
    edge_path: Path,
    run_dir: Path,
    emb_size: int = 16,
    emb_type_list: List[Dict] = None,
    n1_trunc: bool = False
):
    if emb_type_list is None:
        raise ValueError("The parameter 'emb_type_list' must be specified.")

    run_dir.mkdir(parents=True, exist_ok=True)

    for emb_type in emb_type_list:
        rep_type = emb_type["rep_type"]
        regularized = emb_type["regularized"]
        suffix = f"{rep_type}_reg{regularized}"
        output_path = run_dir / f"{suffix}.txt"
        dim = emb_size
        
        if n1_trunc:
            if rep_type == "ULSE-n1":
                dim = emb_size - 1

        use = USE(
            str(edge_path),
            dim,
            str(output_path),
            rep_type=rep_type,
            regularized=regularized
        )

        use.compute_embedding()
        use.save_node_embeddings()