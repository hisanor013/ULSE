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
- Time binning functionality is automatically applied for large-scale datasets (>1000 timesteps)
- Experimental results are saved in `framework/experiments/result/`
