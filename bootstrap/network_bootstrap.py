import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from scipy.sparse import coo_matrix

##### Generate synthetic RDPG graph #####
def generate_latent_positions(n, d):
    """Generate latent positions for each node."""
    return np.random.rand(n, d)

def compute_edge_probabilities(latent_positions):
    """Compute the edge probabilities based on latent positions."""
    n = latent_positions.shape[0]
    P = np.dot(latent_positions, latent_positions.T)
    return P

def generate_rdpg_graph(n, d):
    """Generate an RDPG graph."""
    latent_positions = generate_latent_positions(n, d)
    P = compute_edge_probabilities(latent_positions)
    
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add edges based on probabilities
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < P[i, j]:
                G.add_edge(i, j)
    
    return G

def generate_rdpg_graph_from_latent_positions(latent_positions):
    P = compute_edge_probabilities(latent_positions)
    n = latent_positions.shape[0]
    
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add edges based on probabilities
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < P[i, j] :
                G.add_edge(i, j)
    G = G.to_directed()
    return G

def adjacency_spectral_embedding(adj_matrix, d):
    A = adj_matrix.toarray()
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals)[::-1][:d]  # Select the top d eigenvalues
    embedding = eigvecs[:, idx] * np.sqrt(eigvals[idx])
    return embedding


def network_bootstrap(G, dim = 8,  seed = 1234):
    
    # Convert edge index to a sparse adjacency matrix
    np.random.seed(seed)
    row = G.edge_index[0].numpy()
    col = G.edge_index[1].numpy()
    adj_matrix = coo_matrix((torch.ones(G.edge_index.size(1)), (row, col)), shape=(G.num_nodes, G.num_nodes))
   
    embedding = adjacency_spectral_embedding(adj_matrix, dim)
    # Bootstrap latent positions
    bootstrap_idx = np.random.choice(np.arange(embedding.shape[0]), embedding.shape[0])
    hatA = generate_rdpg_graph_from_latent_positions(embedding[bootstrap_idx,:])
    new_G = from_networkx(hatA)

    # Bootstrap feature
    new_G.x = G.x[bootstrap_idx,:]
    new_G.y = G.y[bootstrap_idx]
    new_G.edge_weight = torch.ones(new_G.edge_index.shape[1])
    return new_G


def generate_network_bootstrap_graphs(graph, n_samples = 10, dim = 8, seed = 1234):
    samples = {}
    for i in np.arange(n_samples):
        G = network_bootstrap(graph, dim = dim, seed = seed+i)
        samples[i] = G
    return samples
        