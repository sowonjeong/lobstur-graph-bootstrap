import numpy as np
import torch
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.data import Data

def train_val_test_split(data, train_percent, split_type='node', seed=1234, device='cpu'):
    np.random.seed(seed)
    # Check if data has labels (y)
    has_labels = data.y is not None and len(data.y) > 0
    
    fold = {}
    # Move the graph data to the specified device (CPU or GPU)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    if data.edge_weight is not None:
        data.edge_weight = data.edge_weight.to(device)
    
    # Edge splitting
    if split_type == 'edge':
        num_edges = data.edge_index.size(1)
        idx = np.random.permutation(num_edges)
        
        # Split indices for training and testing
        train_size = int(num_edges * train_percent)
        train_mask = idx[:train_size]
        test_mask = idx[train_size:]
        
        # Create masks
        train_edges = data.edge_index[:, train_mask]
        test_edges = data.edge_index[:, test_mask]
        
        # Subset edge weights if they exist
        if data.edge_weight is not None:
            train_weights = data.edge_weight[train_mask]
            test_weights = data.edge_weight[test_mask]
        else:
            train_weights, test_weights = None, None

        # Train and test graphs using subgraph
        train_graph = Data(
            x=data.x,
            y=data.y,
            edge_index=train_edges,
            edge_weight=train_weights
        )
        test_graph = Data(
            x=data.x,
            y=data.y,
            edge_index=test_edges,
            edge_weight=test_weights
        )

    # Node splitting
    elif split_type == "node":
        num_nodes = data.x.shape[0]
        idx = np.random.permutation(num_nodes)
        
        train_size = int(num_nodes * train_percent)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        train_mask[idx[:train_size]] = True
        test_mask[idx[train_size:]] = True

        # Convert the boolean masks to node indices
        train_idx = train_mask.nonzero(as_tuple=False).view(-1)
        test_idx = test_mask.nonzero(as_tuple=False).view(-1)

        # Use the subgraph function for faster node filtering
        # train_edges, train_weights = subgraph(train_idx, data.edge_index, data.edge_weight)
        # test_edges, test_weights = subgraph(test_idx, data.edge_index, data.edge_weight)
        train_graph = data.subgraph(train_idx)
        test_graph = data.subgraph(test_idx)
        
        # Create the training and testing graphs with filtered edges and weights
        train_graph.x = data.x[train_idx]
        test_graph.x = data.x[test_idx]
        # test_graph = Data(
        #    x=data.x[test_mask],
        #    edge_index=test_edges,
        #    edge_weight=test_weights
        #)

        # Only include labels if they exist
        if has_labels:
            train_graph.y = data.y[train_mask]
            test_graph.y = data.y[test_mask]
    
    else:
        raise ValueError(f"Unknown split type: {split_type}")


    fold[0] = train_mask
    fold[1] = test_mask
    return train_graph, test_graph, fold


def generate_simple_split_graphs(G, n_samples=50, p=0.7, type="node", seed=1234):
    samples = {}
    for i in range(n_samples):
        samples[i], _, _ = train_val_test_split(G, train_percent=p, split_type=type, seed=seed + i)
    return samples
