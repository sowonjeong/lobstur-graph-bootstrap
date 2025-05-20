import torch
import torch_geometric
from torch_geometric.utils import degree
import networkx as nx
import numpy as np
import random
from collections import Counter
from torch_geometric.utils import from_networkx, to_networkx
from sklearn.neighbors import NearestNeighbors


def preferential_attachment_bootstrap(G, seed=1234):
    random.seed(seed)

    nx_graph = to_networkx(G)

    # Get the degrees of all nodes
    degrees = np.array([deg for _, deg in nx_graph.degree()])

    # Initialize a new graph with a few nodes as the seed
    num_seed_nodes = max(2, int(0.01 * G.num_nodes))  # Start with a few nodes
    initial_nodes = np.random.choice(np.arange(G.num_nodes), size=num_seed_nodes, replace=False)
    
    new_graph = nx.Graph()
    new_graph.add_nodes_from(initial_nodes)
    
    for i in range(num_seed_nodes, G.num_nodes):
        new_node = i

        # Calculate the probability of attaching to each node based on degree
        degree_sum = np.sum(degrees[new_graph.nodes])
        attachment_probs = np.array([degrees[node] / degree_sum for node in new_graph.nodes])

        # Choose a node to attach to, based on the probability distribution
        attach_to_node = np.random.choice(list(new_graph.nodes), p=attachment_probs)

        # Add the new node and connect it to the chosen node
        new_graph.add_node(new_node)
        new_graph.add_edge(new_node, attach_to_node)

    # Convert the new NetworkX graph back to a PyTorch geometric graph
    new_G = from_networkx(new_graph)

    bootstrap_idx = np.random.choice(np.arange(G.num_nodes), new_G.num_nodes, replace=True)
    new_G.x = G.x[bootstrap_idx, :]
    new_G.y = G.y[bootstrap_idx]
    
    new_G.edge_weight = torch.ones(new_G.edge_index.shape[1])
    
    return new_G

def generate_preferential_attachment_bootstrap_graphs(graph, n_samples=10, seed=1234):
    samples = {}
    for i in range(n_samples):
        G = preferential_attachment_bootstrap(graph, seed=seed+i)
        samples[i] = G
    return samples



def precompute_knn(graph, k=20, distance_metric='shortest_path', knn_type='graph'):
    """
    Precompute k-nearest neighbors for a graph.

    Parameters:
        graph: torch_geometric.data.Data or networkx.Graph
            The input graph.
        k: int
            The number of nearest neighbors to compute.
        distance_metric: str
            The metric used for graph-based KNN ('shortest_path', 'shared_neighbors', 'jaccard').
        knn_type: str
            The type of KNN to compute ('graph' or 'feature').

    Returns:
        knn_dict: dict
            A dictionary where keys are node indices and values are lists of k-nearest neighbors.
    """
    # Convert torch_geometric graph to networkx if needed
    if isinstance(graph, torch_geometric.data.Data):
        nx_graph = to_networkx(graph, to_undirected=True)
        node_features = graph.x.cpu().numpy() if hasattr(graph, 'x') else None
    else:
        nx_graph = graph
        node_features = None

    knn_dict = {}

    if knn_type == 'graph':
        # Graph-based KNN
        if distance_metric == 'shortest_path':
            shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph))
            for node in nx_graph.nodes:
                distances = shortest_path_lengths[node]
                sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])
                knn_dict[node] = [n for n, dist in sorted_neighbors if n != node][:k]

        elif distance_metric == 'shared_neighbors':
            for node in nx_graph.nodes:
                neighbors = set(nx_graph.neighbors(node))
                shared_neighbor_counts = {}
                for other_node in nx_graph.nodes:
                    if node != other_node:
                        other_neighbors = set(nx_graph.neighbors(other_node))
                        shared_neighbors = len(neighbors.intersection(other_neighbors))
                        shared_neighbor_counts[other_node] = shared_neighbors
                sorted_neighbors = sorted(shared_neighbor_counts.items(), key=lambda x: -x[1])
                knn_dict[node] = [n for n, count in sorted_neighbors[:k]]

        elif distance_metric == 'jaccard':
            jaccard_similarities = nx.jaccard_coefficient(nx_graph)
            similarity_dict = {}
            for u, v, sim in jaccard_similarities:
                if u not in similarity_dict:
                    similarity_dict[u] = []
                if v not in similarity_dict:
                    similarity_dict[v] = []
                similarity_dict[u].append((v, sim))
                similarity_dict[v].append((u, sim))

            for node in similarity_dict:
                sorted_neighbors = sorted(similarity_dict[node], key=lambda x: -x[1])
                knn_dict[node] = [n for n, sim in sorted_neighbors[:k]]

    elif knn_type == 'feature':
        # Feature-based KNN
        if node_features is None:
            raise ValueError("Feature-based KNN requires node features (graph.x).")

        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(node_features)
        distances, indices = nbrs.kneighbors(node_features)

        for i, neighbors in enumerate(indices):
            knn_dict[i] = neighbors[1:].tolist()  # Exclude self (first neighbor)

    else:
        raise ValueError("Invalid knn_type. Choose 'graph' or 'feature'.")

    return knn_dict



def uniform_bootstrap_edge_rewiring(graph, knn_dict, seed=1234):
    random.seed(seed)

    if isinstance(graph, nx.Graph):
        nx_graph = graph.to_undirected()
    elif isinstance(graph, torch_geometric.data.Data):
        nx_graph = to_networkx(graph, to_undirected=True)
    else:
        raise TypeError("Input must be either a NetworkX graph or a PyTorch Geometric Data object.")
    
    stems = []
    for u, v in nx_graph.edges():
        if u < v:
            stems.extend([u, v])
    
    stem_counts = Counter(stems)
    bootstrapped_graph = nx.Graph()
    bootstrapped_graph.add_nodes_from(nx_graph.nodes()) # make sure the n_node matches

    # Use the precomputed k-NN dictionary for edge rewiring
    while stem_counts:
        u = random.choice(list(stem_counts.keys()))
        stem_counts[u] -= 1
        if stem_counts[u] == 0:
            del stem_counts[u]

        # Get neighbors from the precomputed k-NN dictionary
        available_neighbors = knn_dict[u]
        available_neighbors = [node for node in available_neighbors if node in stem_counts and node != u]

        if available_neighbors:
            new_v = random.choice(available_neighbors)
            bootstrapped_graph.add_edge(u, new_v)
            stem_counts[new_v] -= 1
            if stem_counts[new_v] == 0:
                del stem_counts[new_v]

    return bootstrapped_graph

def weighted_bootstrap_edge_rewiring(graph, knn_dict, seed=1234):
    random.seed(seed)

    # Convert graph to NetworkX format if necessary
    if isinstance(graph, nx.Graph):
        nx_graph = graph.to_undirected()
    elif isinstance(graph, torch_geometric.data.Data):
        nx_graph = to_networkx(graph, to_undirected=True)
    else:
        raise TypeError("Input must be either a NetworkX graph or a PyTorch Geometric Data object.")

    # Create a new bootstrapped graph
    bootstrapped_graph = nx.Graph()
    bootstrapped_graph.add_nodes_from(nx_graph.nodes())  # Ensure node count matches original graph

    # Collect all edges as "stems"
    stem_edges = list(nx_graph.edges())
    stem_counts = Counter(node for edge in stem_edges for node in edge)

    # Main loop for edge rewiring
    while stem_counts:
        # Choose a random node from the stem counts
        node = random.choice(list(stem_counts.keys()))

        # Get neighbors from the precomputed k-NN dictionary
        potential_neighbors = Counter()
        if node not in knn_dict or not knn_dict[node]:
            print(f"Node {node} has no neighbors in knn_dict. Skipping.")
            stem_counts.pop(node, None)  # Remove node if it has no neighbors
            continue

        for u in knn_dict[node]:
            if u not in nx_graph:
                print(f"Node {u} in knn_dict[node] is not in the graph.")
                continue
            for neighbor in nx.neighbors(nx_graph, n=u):
                if (
                    neighbor != node and
                    stem_counts[neighbor] > 0 and
                    neighbor not in nx.neighbors(bootstrapped_graph, n=node)
                ):
                    potential_neighbors[neighbor] += 1

        # Check if there are potential neighbors
        if not potential_neighbors:
            print(f"Node {node} has no potential neighbors. Skipping.")
            stem_counts.pop(node, None)  # Remove node if no potential neighbors
            continue

        # Create a list of neighbors and weights
        neighbors, weights = zip(*potential_neighbors.items())

        # Normalize weights to probabilities
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Select a new neighbor based on probabilities
        new_v = random.choices(neighbors, weights=weights, k=1)[0]
        bootstrapped_graph.add_edge(node, new_v)

        # Update stem counts
        stem_counts[node] -= 1
        stem_counts[new_v] -= 1

        # Remove nodes with zero stem counts
        if stem_counts[node] <= 0:
            stem_counts.pop(node, None)
        if stem_counts[new_v] <= 0:
            stem_counts.pop(new_v, None)

    return bootstrapped_graph

def resample_features(features, knn_dict):
    new_features = np.copy(features)
    for node in knn_dict.keys():
        neighbors = knn_dict[node]
        if neighbors:
            sampled_neighbor = random.choice(neighbors)
            # Resample feature using the neighbor's feature vector
            new_features[node] = features[sampled_neighbor]
    return new_features


def generate_edge_wiring_bootstrap_graphs(graph, n_samples=10, k=20, seed=1234,knn_type = 'graph', distance_metric='jaccard', weighted = True):
    samples = {}
    if knn_type == 'both':
        knn_dict1 = precompute_knn(graph, k=k, distance_metric=distance_metric, knn_type='feature')
        knn_dict2 =  precompute_knn(graph, k=k, distance_metric=distance_metric, knn_type='graph')
    else:
        knn_dict1 = precompute_knn(graph, k=k, distance_metric=distance_metric, knn_type=knn_type)
        knn_dict2 = knn_dict1
    for i in range(n_samples):
        random.seed(seed + i)
        if weighted:
            rewired_graph = weighted_bootstrap_edge_rewiring(graph, knn_dict1, seed=seed + i)
        else:
            rewired_graph = uniform_bootstrap_edge_rewiring(graph, knn_dict1, seed=seed + i)
        G = from_networkx(rewired_graph)
        G.x = torch.tensor(resample_features(graph.x, knn_dict2) , dtype = torch.float32)
        G.y = graph.y
        G.edge_weight = torch.ones(G.edge_index.shape[1])
        samples[i] = G
    return samples


def three_hop_median(graph):
    G = to_networkx(graph, to_undirected=True)
    neighborhood_sizes = []
    
    for node in G.nodes():
        # Get all nodes within 3 hops (distance <= 3)
        three_hop_neighborhood = nx.single_source_shortest_path_length(G, node, cutoff=3)
        
        # The size of the 3-hop neighborhood (number of nodes)
        neighborhood_size = len(three_hop_neighborhood)
        
        # Append the size to the list
        neighborhood_sizes.append(neighborhood_size)
    
    # Compute the median of all 3-hop neighborhood sizes
    median_size = np.median(neighborhood_sizes)
    
    return median_size
