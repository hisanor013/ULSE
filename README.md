# ULSE (Unfolded Laplacian Spectral Embedding) Framework

## Overview

This project provides implementations and benchmarks for various embedding methods on dynamic graphs (time-varying graphs). It supports a wide range of methods from traditional non-deep learning approaches to state-of-the-art deep learning techniques.

## Key Features

### 1. Embedding Method Implementations

`framework/main.py` allows you to execute the following non-deep learning embedding methods:

- **USE (Unfolded Spectral Embedding)**
  - UASE: Unfolded Adjacency Spectral Embedding
  - ULSE-n1: Unfolded Laplacian Spectral Embedding (n1)
  - ULSE-n2: Unfolded Laplacian Spectral Embedding (n2)

- **TemporalCut**: Spectral cut-based temporal graph embedding

- **Node2Vec**: Random walk-based embedding for static graphs

- **OMNI**: Omnibus embedding method integrating all time points

### 2. Deep Learning Methods

For deep learning-based methods (DyRep, etc.), please refer to the corresponding directories. Experimental results include the following methods:
- DyRep
- DyGFormer
- JODIE
- TGN

## Data Preprocessing

### School Dataset
When using the `school` dataset, apply the following preprocessing script beforehand:

```bash
python convert_school_time.py
```

This script converts timestamps into discrete intervals (1-5) and normalizes the data.

## Experiments and Plotting

### Synthetic Data and Embedding Visualization

For generating synthetic data and plotting embeddings, use the following Jupyter Notebook:

```
synthetic_data/main_UASE_vs_ULSE.ipynb
```

This notebook provides:
- Synthetic data generation
- Comparison of various embedding methods
- Result visualization and plotting

## Usage

### Basic Usage

```bash
# Example of running USE
python framework/main.py --dataset brain --method USE --rep_type ULSE-n1

# Run all methods
python framework/main.py --dataset brain --method all

# Example of running TemporalCut
python framework/main.py --dataset school --method TemporalCut
```

### Datasets

Available datasets:
- `synthetic_1`, `synthetic_2`: Synthetic data
- `school`: School contact network (requires preprocessing)
- `brain`: Brain network
- `stock`: Stock correlation network

### Parameters

For detailed parameters, see:
```bash
python framework/main.py --help
```

### Clustering Evaluation and Comparison Tables

After computing embeddings using `framework/main.py`, you can evaluate the clustering performance and generate comparison tables (CSV and LaTeX formats) using the clustering evaluation script.

#### Step 1: Compute Embeddings

First, compute embeddings for the methods you want to compare:

```bash
# Compute embeddings for all methods
python framework/main.py --dataset brain --method all

# Or compute embeddings for specific methods
python framework/main.py --dataset brain --method USE --rep_type ULSE-n1
python framework/main.py --dataset brain --method TemporalCut
```

The embeddings will be saved in `emb/{dataset}/` directory.

#### Step 2: Run Clustering Evaluation

After embeddings are computed, run the clustering evaluation script to generate comparison tables:

```bash
# Evaluate clustering for a single dataset
python framework/experiments/clustering.py --dataset brain

# Evaluate clustering for all datasets
python framework/experiments/clustering.py --dataset all

# Specify custom output path
python framework/experiments/clustering.py --dataset brain --output result/my_results.csv
```

#### Output Files

The clustering evaluation script generates the following files in the `framework/experiments/result/` directory (or the specified output directory):

1. **Detailed Results CSV** (`clustering_results_{dataset}.csv`):
   - Contains detailed evaluation results for each method and dataset
   - Includes ACC, NMI, ARI, F1 scores, number of nodes, clusters, and time steps

2. **Summary CSV** (`clustering_summary_{dataset}.csv`):
   - Contains average performance metrics across datasets for each method

3. **Comparison Table CSV** (`clustering_table_{dataset}.csv`):
   - Contains formatted comparison table with best results highlighted
   - Organized by dataset and metric (ACC, NMI, ARI, F1)

4. **LaTeX Table** (`clustering_table_{dataset}.tex`):
   - LaTeX-formatted comparison table ready for paper inclusion
   - Best results are bolded, second-best results are underlined

5. **LaTeX Summary** (`clustering_summary_{dataset}.tex`):
   - LaTeX-formatted summary table with average performance

The script also displays a formatted comparison table in the terminal, showing which methods perform best for each metric.

## Directory Structure

```
├── framework/           # Main embedding method implementations
│   ├── main.py         # Entry point
│   ├── model/          # Implementation of each method
│   └── experiments/    # Experimental results
├── synthetic_data/     # Synthetic data generation and plotting
├── data/              # Datasets
├── convert_school_time.py  # School data preprocessing
└── emb/               # Embedding results storage
```

## Requirements

```bash
pip install -r requirements.txt
```

## Notes

- For deep learning methods (DyRep, etc.), please refer to the implementations in the corresponding directories
- Time binning functionality can be enabled with `--use_time_binning` flag. When enabled, datasets with more timesteps than `--max_timesteps` (default: 100) will be automatically binned
- Experimental results are saved in `framework/experiments/result/`
