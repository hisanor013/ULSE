from pathlib import Path
from typing import List, Tuple
from embedding.compute import compute_embeddings
from embedding.cluster import cluster_embeddings_kmeans_and_save

def run_embedding_and_prediction_pipeline(
    base_dir: Path,
    pq_list: List[Tuple[float, float]],
    runs_per_pq: int,
    clusters_per_time: List[int],
    emb_size: int = 16,
    random_state: int = 42,
    emb_type_list: List[dict] = None,
    emb_dir: Path = None
):
    if emb_type_list is None:
        raise ValueError("The parameter 'emb_type_list' must be specified.")
    if emb_dir is None:
        emb_dir = base_dir

    for p, q in pq_list:
        for run_id in range(runs_per_pq):
            run_dir = base_dir / f"p{p}_q{q}" / f"run{run_id}"
            edge_path = run_dir / "synthetic.txt"

            if not edge_path.exists():
                print(f"⚠️ File 'synthetic.txt' not found at: {edge_path}")
                continue

            print(f"⏳ Performing embedding: p={p}, q={q}, run={run_id}")

            emb_run_dir = emb_dir / f"p{p}_q{q}" / f"run{run_id}"
            emb_run_dir.mkdir(parents=True, exist_ok=True)
            
            compute_embeddings(
                edge_path=edge_path,
                run_dir=emb_run_dir,
                emb_size=emb_size,
                emb_type_list=emb_type_list
            )

            for emb_type in emb_type_list:
                rep_type = emb_type["rep_type"]
                regularized = emb_type["regularized"]
                emb_name = f"{rep_type}_reg{regularized}"

                embedding_path = emb_run_dir / f"{emb_name}.txt"
                output_path = emb_run_dir / f"predicted_labels_{emb_name}.txt"

                if not embedding_path.exists():
                    print(f"⚠️ Could not find the embedding file at: {embedding_path}")
                    continue

                print(f"⏳ Clustering: {embedding_path.name}")
                
                cluster_embeddings_kmeans_and_save(
                    embedding_path=embedding_path,
                    clusters_per_time=clusters_per_time,
                    output_path=output_path,
                    random_state=random_state,
                )
                
                print(f"✅ Clustering completed: {output_path.name}")

