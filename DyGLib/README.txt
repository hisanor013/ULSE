Temporal Graph Embedding Pipeline (DyGLib-Compatible)
====================================================

Overview
--------

This repository provides reproducible pipeline for processing dynamic graph data, training temporal graph neural networks (GNNs), and extracting time-aware node embeddings. 

To use this pipeline, download the DyGLib source code and place all folders here:

https://github.com/yule-BUAA/DyGLib

We thank the DyGLib authors for making their code publicly available.

Pipeline Overview
-----------------

The workflow consists of the following three sequential steps:

### Step 1: Data Preparation  
Notebook: `step1_prepare_data_for_dyglib.ipynb`  
- Converts raw temporal edge lists into DyGLib-compatible input formats.  

### Step 2: Model Training 
Notebook: `step2_learn_embedding.ipynb`  
- Trains a temporal GNN model on the processed dataset.  
- Supports various architectures (e.g., TGN, DyGFormer).  
- Outputs trained model checkpoints and training logs.

### Step 3: Node Embedding Extraction  
Notebook: `step3_obtain_node_embeddings_from_trained_models.ipynb`  
- Loads the trained model from Step 2.  
- Discretizes timestamps and groups interactions.  
- Computes temporal node embeddings per time step.  
- Saves embeddings to text and pickle files.

Step-by-Step Details
--------------------

### Step 1: Data Preparation

**Notebook**: `step1_prepare_data_for_dyglib.ipynb`

**Input Format**:
<source_node> <target_node> <timestamp>

**Main Operations**:
- Load edge lists (e.g., `brain.txt`, `school.txt`, `stock.txt`)
- Add reverse edges to simulate undirected graphs
- Remove exact duplicate edges
- Generate synthetic edge/node features (random or zero-filled)

**Output Directory**: `./processed_data/<dataset_name>/`

- `ml_<dataset_name>.csv` : Processed edge list  
- `ml_<dataset_name>.npy` : Edge feature matrix  
- `ml_<dataset_name>_node.npy` : Node feature matrix

---

### Step 2: Model Training via Link Prediction

**Notebook**: `step2_learn_embedding_through_link_prediction.ipynb`

**Supported Models**:
- JODIE  
- DyRep  
- TGN  
- DyGFormer  

**Output**:

- Trained model checkpoint:  
  `./saved_models/<model_name>/<dataset_name>/<model_name>_seed<seed>.pkl`

- Training logs:  
  `./logs/<model_name>/<dataset_name>/<model_name>_seed<seed>/...`

---

### Step 3: Node Embedding Extraction

**Notebook**: `step3_obtain_node_embeddings_from_trained_models.ipynb`

**Operations**:
- Load the trained model checkpoint from Step 2  
- Discretize time and group interactions  
- Compute embeddings for nodes at each time step  

**Output Files**:
- Pickle:  
  `node_embeddings_discrete_avg_<model_name>_<dataset_name>.pkl`

- Text:  
  `node_embeddings_discrete_avg_<model_name>_<dataset_name>.txt`

**Text File Format**:
<num_node-time_pairs> <num_timesteps>
<node_id> <embedding_1> <embedding_2> ... <embedding_d>
...

Dependencies
------------

- Python 3.8 or higher  
- PyTorch  
- numpy  
- pandas  
- tqdm  
- Jupyter Notebook  
- DyGLib (or compatible implementation)

Usage Instructions
------------------

1. **Data Preparation**  
   Modify `input_edgelist_path` and `dataset_name` as needed.  
   Run `step1_prepare_data_for_dyglib.ipynb`.

2. **Model Training**  
   Set your desired `model_name` and match the `dataset_name`.  
   Run `step2_learn_embedding_through_link_prediction.ipynb`.

3. **Embedding Extraction**  
   Using the same `model_name` and `dataset_name`, run `step3_obtain_node_embeddings_from_trained_models.ipynb`.

4. Move files to ../emb
   Please move these text files to the ../emb folder so we can compare them with other methods.
   
> All notebooks should be executed sequentially.  
> A GPU is strongly recommended for Steps 2 and 3.

