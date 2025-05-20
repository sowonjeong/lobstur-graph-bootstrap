import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from scipy import stats
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import math

from bootstrap.block_bootstrap import *
from bootstrap.np_bootstrap import *
from bootstrap.network_bootstrap import *
from bootstrap.vae_sample import *


def compute_graph_statistics(graph):
    if isinstance(graph, nx.Graph):
        G = graph
    elif isinstance(graph, Data):
        G = to_networkx(graph, to_undirected=True)
    else:
        raise TypeError("Input must be a NetworkX graph or a torch_geometric Data object.")
    
    # Compute core statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
   # try:
   #     eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
   # except nx.PowerIterationFailedConvergence:
   #     print("Eigenvector centrality did not converge for this graph.")
   #     eigenvector_centrality = None

    # Node-level statistics
    node_level_stats = {
        "degree_distribution": [d for n, d in G.degree()],
        "clustering_coefficient": nx.clustering(G),
        # "betweenness_centrality": nx.betweenness_centrality(G),
        # "eigenvector_centrality": eigenvector_centrality,
        # "closeness_centrality": nx.closeness_centrality(G),
        "pagerank": nx.pagerank(G),
        # "eccentricity": nx.eccentricity(G) if nx.is_connected(G) else None,
        "num_triangles": nx.triangles(G).values()      
    }

    # Graph-level statistics
    graph_level_stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": nx.density(G),
        "avg_clustering_coefficient": nx.average_clustering(G),
        "avg_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan,
        "diameter": nx.diameter(G) if nx.is_connected(G) else np.nan,
        "radius": nx.radius(G) if nx.is_connected(G) else np.nan,
        "num_connected_components": nx.number_connected_components(G),
        "giant_component_size": len(max(nx.connected_components(G), key=len)),
        # "modularity": nx.algorithms.community.quality.modularity(G, nx.algorithms.community.greedy_modularity_communities(G)),
        "assortativity": nx.degree_assortativity_coefficient(G),
        # "betweenness_centrality": np.mean(list(node_level_stats['betweenness_centrality'])),
        # "eigenvector_centrality": np.mean(list(node_level_stats['eigenvector_centrality'])) if nx.is_connected(G) else np.nan,
        # "closeness_centrality": np.mean(list(node_level_stats['closeness_centrality'])),
        "pagerank": np.mean(list(node_level_stats['pagerank'])),
        "transitivity": nx.transitivity(G),
        "num_triangles": sum(node_level_stats['num_triangles']) // 3  # Each triangle is counted three times
    }

    return node_level_stats, graph_level_stats


def hellinger(P, Q):
    return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2)) / np.sqrt(2)

def total_variation_distance(P, Q):
    return 0.5 * np.sum(np.abs(P - Q))


def metric_entropy(data1, data2):
    res = {}
    # Create histograms (discretize the distributions)
    hist1, _ = np.histogram(data1, bins=50, density=True)
    hist2, _ = np.histogram(data2, bins=50, density=True)

    # Fit KDE to the empirical data
    kde1 = KernelDensity(bandwidth=0.5).fit(np.array(data1).reshape(-1,1))
    kde2 = KernelDensity(bandwidth=0.5).fit(np.array(data2).reshape(-1,1))

    # Evaluate KDEs on a grid of points
    x_grid = np.linspace(min(np.array(data1).min(), np.array(data2).min()), max(np.array(data1).max(), np.array(data2).max()), 1000).reshape(-1, 1)
    log_density1 = kde1.score_samples(x_grid)
    log_density2 = kde2.score_samples(x_grid)

    # Convert log densities to probabilities
    density1 = np.exp(log_density1)
    density2 = np.exp(log_density2)

    # Calculate KL Divergence
    res['kl_divergence'] = entropy(hist1, hist2)
    res['jensenshannon']= jensenshannon(hist1, hist2)
    res['wasserstein'] = wasserstein_distance(data1, data2)
    res['hellinger'] = hellinger(density1, density2)
    res['TV'] =total_variation_distance(density1, density2)
    return res



def extract_data(A, key):
    if key == 'degree_distribution':
        data1 = A[key]
    elif key == 'num_triangles':
        data1 = list(A[key])
    else:
        data_temp = A[key]
        if data_temp is None:
            print("Warning: data_temp is None.")
            return []
        data1 = [data_temp[node] for node in data_temp]
    return data1

def node_ref_table(org_node, node_stat, node_stat_ref, type = 'kl_divergence'):
    kl_list = {}
    n_samples = len(node_stat.keys())
    valid_keys = [key for key in org_node.keys()] #if isinstance(A_node[key], (list, np.ndarray)) and len(A_node[key]) > 0]    
    for idx, key in enumerate(valid_keys):
        p_values = []
        data1 = extract_data(org_node, key)
        for i in np.arange(n_samples):
            data2 = extract_data(node_stat[i],key)
            data3 = extract_data(node_stat_ref[i], key)
            kl_sample = metric_entropy(data3, data2)[type]
            kl_real = metric_entropy(data3, data1)[type]
            kl_list[key+'_sample_'+str(i)]={'within_sample': kl_sample,
                                            'to_original':kl_real,
                                            'key':key, 
                                            'sample':i}
    return pd.DataFrame(kl_list).T

def plot_node_stat_table(kl_df):
    # Get unique keys
    unique_keys = kl_df['key'].unique()

    # Set up the subplots: 5 keys, so 5 subplots side by side
    fig, axs = plt.subplots(1, len(unique_keys), figsize=(20, 5), sharey=True)

    # Loop through each unique key and corresponding subplot axis
    for i, key in enumerate(unique_keys):
        # Filter the DataFrame by key
        df_key = kl_df[kl_df['key'] == key]
        df_key = df_key.copy()
        # Convert columns to numeric, forcing non-numeric values to NaN
        df_key['within_sample'] = pd.to_numeric(df_key['within_sample'], errors='coerce')
        # df_key['within_sample'] = pd.to_numeric(df_key['within_sample'], errors='coerce')
        df_key['to_original'] = pd.to_numeric(df_key['to_original'], errors='coerce')
        #df_key['to_original'] = pd.to_numeric(df_key['to_original'], errors='coerce')

        # Replace infinite values with NaN
        inf_count_to_original = np.isinf(df_key['to_original']).sum()
        df_key.replace([np.inf, -np.inf], np.nan)
        df_key = df_key.dropna(subset=['within_sample', 'to_original']).copy()
        
        # Overlay histograms on the corresponding subplot axis
        axs[i].hist(df_key['within_sample'], bins=10, alpha=0.5, label='within_sample')
        axs[i].hist(df_key['to_original'], bins=10, alpha=0.5, label='to_original')
        
        # Set titles and labels for each subplot
        axs[i].set_title(f'{key} (inf count: {inf_count_to_original})')
        axs[i].set_xlabel('Value')
        axs[i].grid(True)
        
        if i == 0:  # Only add y-label to the first subplot
            axs[i].set_ylabel('Frequency')

    # Add a single legend for all subplots
    axs[0].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



def ks_test_and_plot_all_keys(A_node, bb_node, num_cols=5):
    """
    A_node: node-level statistics for the original graph
    bb_node: node-level statistics for the generated graphs
    """
    # Get valid keys (excluding those with NaN or invalid data in A_node)
    valid_keys = [key for key in A_node.keys()] #if isinstance(A_node[key], (list, np.ndarray)) and len(A_node[key]) > 0]
    
    # Calculate the total number of plots
    total_plots = len(valid_keys)
    
    # Calculate the number of rows needed
    num_rows = (total_plots // num_cols) + (total_plots % num_cols > 0)
    
    # Set up the figure for subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axes = axes.flatten()  # Flatten in case we have a grid of subplots
    
    for idx, key in enumerate(valid_keys):
        p_values = []
        data1 = extract_data(A_node, key)
        for i in np.arange(50):
            data2 = extract_data(bb_node[i],key)
            _, p_value = stats.ks_2samp(data1, data2)
            p_values.append(p_value)

        if p_values:
            # Sort the p-values
            p_values.sort()

            # Plot sorted p-values
            axes[idx].plot(p_values)
            axes[idx].set_title(f'P-values for {key}')
            axes[idx].set_xlabel('Sample index')
            axes[idx].set_ylabel('P-value')
            axes[idx].grid(True)

    # Hide any unused subplots
    # for i in range(len(valid_keys), len(axes)):
    #    fig.delaxes(axes[i])
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def quantile_in_bootstrap(orig_stat, sampled_stat, names, max_plots_per_row=4):
    # Number of subplots and rows needed
    num_plots = len(names)
    num_rows = (num_plots + max_plots_per_row - 1) // max_plots_per_row  # Ceiling division for rows
    
    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(num_rows, min(max_plots_per_row, num_plots), figsize=(6* max_plots_per_row, 6 * num_rows))
    axes = np.atleast_2d(axes)  # Ensure axes is a 2D array even if there's only one row
    
    # Flatten axes for easy indexing if necessary
    axes = axes.flatten()

    # Loop over the names and plot each
    results = {}
    for i, name in enumerate(names):
        if not math.isnan(orig_stat[name]):
            # Assuming bb_graph is your dictionary
            num_nodes_values = np.array([sampled_stat[j][name] for j in np.arange(50)])
            
            # Given value
            given_value = orig_stat[name]
            
            # Plotting the histogram on the appropriate axis
            axes[i].hist(num_nodes_values, bins=10, color='blue', alpha=0.7)
            
            # Add a red vertical line for the given value
            axes[i].axvline(given_value, color='red', linestyle='dashed', linewidth=2)
            
            # Calculate the quantiles
            quantiles = np.quantile(num_nodes_values, np.arange(0.0, 1.01, 0.01))  # 100 quantiles (percentiles)
            
            # Determine the quantile of the given value
            quantile_index = np.searchsorted(quantiles, given_value, side='right') - 1
            
            # Adjust for edge cases
            if quantile_index < 0:
                quantile_index = 0
            elif quantile_index >= len(quantiles) - 1:
                quantile_index = len(quantiles) - 2
            
            # Add labels and title to the subplot
            axes[i].set_title(f"{name}: Quantile {quantile_index + 1} out of 100")
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Frequency')
            
            # Add a label for the given value
            axes[i].text(given_value, axes[i].get_ylim()[1] * 0.5, f'{given_value:.3f}', color='red', rotation=90)
            
            # Grid
            axes[i].grid(True)
            
            # Store the result for this key
            results[name] = (given_value, quantile_index, quantiles)
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    return results


def gstat_absolute_val_in_bootstrap(orig_stat, sampled_stat, names, max_plots_per_row=4, output_dir="plots"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Number of subplots and rows needed
    num_plots = len(names)
    num_rows = (num_plots + max_plots_per_row - 1) // max_plots_per_row  # Ceiling division for rows

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(num_rows, min(max_plots_per_row, num_plots), figsize=(6 * max_plots_per_row, 6 * num_rows))
    axes = np.atleast_2d(axes)  # Ensure axes is a 2D array even if there's only one row

    # Flatten axes for easy indexing if necessary
    axes = axes.flatten()
    n_samples = len(sampled_stat.keys())

    # Results table
    results = []

    for i, name in enumerate(names):
        if not math.isnan(orig_stat[name]):
            # Extract the sampled statistics for the current name
            num_nodes_values = np.array([sampled_stat[j][name] for j in np.arange(n_samples)])

            # Given value
            given_value = orig_stat[name]

            # Calculate mean and standard deviation
            mean_value = np.mean(num_nodes_values)
            std_value = np.std(num_nodes_values)
            median_value = np.median(num_nodes_values)
            quantile_25 = np.percentile(num_nodes_values, 25)
            quantile_75 = np.percentile(num_nodes_values, 75)

            # Plotting the histogram on the appropriate axis
            axes[i].hist(num_nodes_values, bins=10, color='blue', alpha=0.7)

            # Add a red vertical line for the given value
            axes[i].axvline(given_value, color='red', linestyle='dashed', linewidth=2)

            # Add labels and title to the subplot
            axes[i].set_title(f"{name}: Mean={mean_value:.2f}, SD={std_value:.2f}")
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Frequency')

            # Add a label for the given value
            axes[i].text(given_value, axes[i].get_ylim()[1] * 0.5, f'{given_value:.3f}', color='red', rotation=90)

            # Grid
            axes[i].grid(True)

            # Save the result for this statistic
            results.append({
                'Statistic': name,
                'Orig_Stat_Value': given_value,
                'Mean': mean_value,
                'SD': std_value,
                'Median': median_value,
                '25th_Quantile': quantile_25,
                '75th_Quantile': quantile_75
            })

            # Save the individual plot
            plt.figure()
            plt.hist(num_nodes_values, bins=10, color='blue', alpha=0.7)
            plt.axvline(given_value, color='red', linestyle='dashed', linewidth=2)
            plt.title(f"{name}: Mean={mean_value:.2f}, SD={std_value:.2f}")
            plt.xlabel(name)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{name}_histogram.png"))
            plt.close()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)

def summarize_graph_statistics(*outputs, method_names):
    """
    Summarize graph statistics from multiple methods into a single DataFrame.

    Args:
        outputs: A list of pandas DataFrames from gstat_absolute_val_in_bootstrap.
        method_names: A list of method names corresponding to the outputs.
    
    Returns:
        summary_df: A unified pandas DataFrame summarizing the statistics.
    """
    summary_list = []

    for method_name, output in zip(method_names, outputs):
        output['Method'] = method_name  # Add a column to identify the method
        summary_list.append(output)

    # Concatenate all outputs into a single DataFrame
    summary_df = pd.concat(summary_list, ignore_index=True)
    return summary_df



## compare all at once and get the result ## 
def compare_statistics(graph, sample, sample_ref, div_type = "jensenshannon", output_dir='None'):
    c_node, c_graph = compute_graph_statistics(graph)

    sample_node = {}
    sample_graph = {}
    n_samples = len(sample.keys())
    for i in np.arange(n_samples):
        sample_node[i], sample_graph[i] = compute_graph_statistics(sample[i])
    
    sample_node_ref = {}
    sample_graph_ref = {}
    for i in np.arange(n_samples):
        sample_node_ref[i], sample_graph_ref[i] = compute_graph_statistics(sample_ref[i])

    ### Graph Level Comparison
    non_nan_keys = [key for key in sample_graph[0] if not math.isnan(sample_graph[0][key])]
    results =  gstat_absolute_val_in_bootstrap(c_graph, sample_graph, non_nan_keys,output_dir = output_dir)

    ### Node Level Comparison
    nb_tab = node_ref_table(c_node, sample_node, sample_node_ref, type= div_type)
    plot_node_stat_table(nb_tab)
    
    return results, nb_tab

def run_ks_test(data1, data2):
    # Perform the KS test
    ks_statistic, p_value = stats.ks_2samp(data1, data2)
    return ks_statistic, p_value


def run_node_stat_compare(df):
    results = []
    
    # Group by the 'key' column
    grouped = df.groupby('key')
    
    # Run KS test and metric entropy for each group
    for key, group in grouped:
        data1 = group['within_sample']
        data2 = group['to_original']
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(data1, data2)
        
        # Perform metric entropy calculations
        metrics = metric_entropy(data1, data2)
        
        # Store the result in a dictionary
        result = {
            'key': key,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            **metrics  # Add all metrics from metric_entropy result
        }
        
        results.append(result)
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def extract_quantiles(dict_list):
    # List of dictionaries
    # dict_list = [dict1, dict2, dict3]

    # Extract b_values from each dictionary and store them in a list
    b_values = [{k: v[1] for k, v in d.items()} for d in dict_list]

    # Convert to a pandas DataFrame
    return pd.DataFrame(b_values)

def run_graph_stat_compare(graph, n_samples = 1000, type = "wasserstein", figure_path = None):
    d = graph.x.shape[1]
    p = np.minimum(d, 8)
    ed_sample = generate_simple_split_graphs(graph, n_samples = n_samples, p = 0.8, type = "edge", seed = 1234)
    ed_sample_ref = generate_simple_split_graphs(graph, n_samples =  n_samples, p = 0.8, type = "edge", seed = 5678)
    print("ed sample generated!")

    nd_sample = generate_simple_split_graphs(graph, n_samples =  n_samples, p = 0.8, type = "node", seed = 1234)
    nd_sample_ref = generate_simple_split_graphs(graph, n_samples =  n_samples, p = 0.8, type = "node", seed = 5678)
    print("nd sample generated!")

    nb_sample = generate_network_bootstrap_graphs(graph,n_samples =  n_samples, dim = p, seed = 1)
    nb_sample_ref = generate_network_bootstrap_graphs(graph,n_samples =  n_samples, dim = p, seed = 1+n_samples)
    print("nb sample generated!")

    er_sample = generate_edge_wiring_bootstrap_graphs(graph,n_samples =  n_samples, seed = 1)
    er_sample_ref = generate_edge_wiring_bootstrap_graphs(graph,n_samples =  n_samples, seed = 1+n_samples)
    print("er sample generated!")

    vae_sample = generate_vae_samples(graph,  n_samples, graph.x.shape[1], 16, p, verbose = False)
    vae_sample_ref = generate_vae_samples(graph,  n_samples, graph.x.shape[1], 16, p, verbose = False)
    print("vae sample generated!")

    ed_out, ed_tab = compare_statistics(graph, ed_sample, ed_sample_ref, output_dir = figure_path+'/ed')
    print("ed stats computed!")

    nd_out, nd_tab = compare_statistics(graph, nd_sample, nd_sample_ref, output_dir = figure_path+'/nd')
    print("nd stats computed!")

    nb_out, nb_tab = compare_statistics(graph, nb_sample, nb_sample_ref, output_dir = figure_path+'/nb')
    print("nb stats computed!")

    er_out, er_tab = compare_statistics(graph, er_sample, er_sample_ref, output_dir = figure_path+'/er')
    print("er stats computed!")
    vae_out, vae_tab = compare_statistics(graph, vae_sample, vae_sample_ref, output_dir = figure_path+'/vae')
    print("vae stats computed!")

    tables = {'ed':run_node_stat_compare(ed_tab),
              'nd': run_node_stat_compare(nd_tab),
              'nb': run_node_stat_compare(nb_tab),
               'er': run_node_stat_compare(er_tab),
                'vae':run_node_stat_compare(vae_tab)}
    n_stat = pd.DataFrame({
    table_name: table[type] for table_name, table in tables.items()})
    g_stat = summarize_graph_statistics(ed_out, nd_out, nb_out, er_out, vae_out, method_names=['ed', 'nd', 'nb', 'er', 'vae'])
    return n_stat, g_stat


def run_graph_stat_compare_er(graph, n_samples = 1000, k = 20, distance_metric='shortest_path', knn_type='graph', weighted = True, figure_path = None):

    er_sample = generate_edge_wiring_bootstrap_graphs(graph,n_samples =  n_samples, seed = 1, k = k, knn_type=knn_type, distance_metric = distance_metric, weighted = weighted)
    er_sample_ref = generate_edge_wiring_bootstrap_graphs(graph,n_samples =  n_samples, seed = 1+n_samples, k = k,  knn_type=knn_type, distance_metric=distance_metric, weighted= weighted)
    er_out, er_tab = compare_statistics(graph, er_sample, er_sample_ref, output_dir = figure_path+'/er')

    n_stat = run_node_stat_compare(er_tab)
    g_stat = er_out
    return n_stat, g_stat

def run_graph_stat_compare_er_simple(graph, n_samples = 1000, k = 20, distance_metric='jaccard', knn_type='graph',figure_path = None):
    c_node, c_graph = compute_graph_statistics(graph)
    sample = generate_edge_wiring_bootstrap_graphs(graph,n_samples =  n_samples, seed = 1, k = k, knn_type=knn_type, distance_metric = distance_metric)
    sample_graph ={}
    n_samples = len(sample.keys())
    for i in np.arange(n_samples):
        _, sample_graph[i] = compute_graph_statistics(sample[i])
    non_nan_keys = [key for key in sample_graph[0] if not math.isnan(sample_graph[0][key])]
    results =  gstat_absolute_val_in_bootstrap(c_graph, sample_graph, non_nan_keys,output_dir = figure_path)
    return results