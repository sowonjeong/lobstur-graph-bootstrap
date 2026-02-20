# 🦞 LOBSTUR: Graph Bootstrap

**LOBSTUR** (**L**ocal **B**ootstrap for Tuning **U**nsupervised **R**epresentations) is a framework for selecting hyperparameters of Graph Neural Networks *without labeled data*, using graph-aware bootstrap resampling.

> **Paper:** Jeong, S. & Donnat, C. (2025). *LOBSTUR: A Local Bootstrap Framework for Tuning Unsupervised Representations in Graph Neural Networks.* NeurIPS 2025 Workshop on New Frontiers in Graph and Geometric Machine Learning (NPGML). [[OpenReview]](https://openreview.net/forum?id=5Q1F4ovQG1)

---

## Overview

A central challenge in deploying unsupervised GNNs is hyperparameter selection: without labels, there is no natural validation signal.

LOBSTUR addresses this by:

1. **Generating bootstrap replicates** of the input graph that respect its local structure
2. **Training the GNN independently** on pairs of replicates
3. **Measuring embedding consistency** across replicate pairs via Canonical Correlation Analysis (CCA)
4. **Selecting hyperparameters** that maximize this consistency — a fully unsupervised criterion

Compared to an uninformed hyperparameter choice, LOBSTUR achieves up to a **65.9% improvement in classification accuracy** on standard benchmarks.

---

## Repository Structure

```
lobstur-graph-bootstrap/
├── bootstrap/                   # Graph bootstrap methods
│   ├── __init__.py
│   ├── network_bootstrap.py     # Spectral / RDPG bootstrap
│   ├── np_bootstrap.py          # Nonparametric edge-rewiring bootstrap  ← main method
│   ├── vae_sample.py            # Variational Graph Autoencoder sampling
│   └── simple_split.py         # Node / edge split baselines
├── eval/                        # Evaluation utilities
│   ├── __init__.py
│   ├── evaluation.py            # Embedding metrics (CCA distance, NKR, clustering)
│   └── graph_statistics.py      # Graph-level statistics and distribution tests
└── requirements.txt
```

---

## Bootstrap Methods

| Method | Module | Description |
|--------|--------|-------------|
| **Network Bootstrap** | `bootstrap/network_bootstrap.py` | Computes Adjacency Spectral Embedding (ASE), bootstraps the latent positions, and regenerates a graph via the Random Dot Product Graph (RDPG) model. |
| **Edge-Rewiring Bootstrap** | `bootstrap/np_bootstrap.py` | Nonparametric: rewires edges within a KNN neighbourhood (graph- or feature-based), then resamples node features from nearby nodes. Uniform and degree-weighted variants are provided. |
| **VAE Sampling** | `bootstrap/vae_sample.py` | Fits a Variational Graph Autoencoder on the input and draws new graphs from the learned latent distribution. |
| **Node / Edge Split** | `bootstrap/simple_split.py` | Simple subsampling baselines that hold out a fraction of nodes or edges. |

---

## Installation

```bash
git clone https://github.com/<your-org>/lobstur-graph-bootstrap.git
cd lobstur-graph-bootstrap
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- NetworkX, NumPy, SciPy, scikit-learn, pandas, matplotlib

---

## Quick Start

```python
import torch
from torch_geometric.datasets import Planetoid

# Load data
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
data.edge_weight = torch.ones(data.edge_index.shape[1])

# --- Generate bootstrap samples ---
from bootstrap.network_bootstrap import generate_network_bootstrap_graphs
from bootstrap.np_bootstrap import generate_edge_wiring_bootstrap_graphs

nb_samples = generate_network_bootstrap_graphs(data, n_samples=20, dim=8, seed=1)
er_samples = generate_edge_wiring_bootstrap_graphs(
    data, n_samples=20, k=20, seed=1,
    knn_type='feature', distance_metric='jaccard', weighted=True
)

# --- Evaluate embedding consistency ---
from eval.evaluation import cca_dist

# Train your GNN on two replicates, then:
# z1 = your_model(er_samples[0])
# z2 = your_model(er_samples[1])
# consistency = cca_dist(z1, z2)   # lower = more consistent
```


---

## Evaluation Metrics

`eval/evaluation.py` provides model-agnostic metrics that work on any embedding matrix:

| Metric | Function | Description |
|--------|----------|-------------|
| **CCA Distance** | `cca_dist(z1, z2)` | Frobenius norm between CCA-aligned embeddings — the primary LOBSTUR criterion |
| **Neighbor Kept Ratio** | `neighbor_kept_ratio_eval(z1, z2)` | Fraction of KNN neighbours preserved across replicates |
| **Linear Classifier Accuracy** | `linear_classifier(z, labels)` | Downstream classification accuracy (requires labels; used for evaluation only) |
| **Full CV Suite** | `evaluate_CV(z1, z2, z_out, labels)` | KMeans/GMM ARI & NMI, hierarchical clustering Spearman, NKR, CCA distance |

`eval/graph_statistics.py` provides tools to verify that bootstrap samples faithfully preserve the original graph's structure (degree distribution, clustering coefficient, assortativity, etc.) using KL divergence, Wasserstein, Hellinger, and Jensen–Shannon distances.

---

## Citation

If you use LOBSTUR in your research, please cite:

```bibtex
@inproceedings{jeong2025lobstur,
  title     = {LOBSTUR: A Local Bootstrap Framework for Tuning Unsupervised Representations in Graph Neural Networks},
  author    = {Jeong, Sowon and Donnat, Claire},
  booktitle = {NeurIPS 2025 Workshop on New Frontiers in Graph and Geometric Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=5Q1F4ovQG1}
}
```

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
