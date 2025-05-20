import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from utils.graph_utils import *


def create_grid(sample_df, x_col, y_col, max_x = 11044, max_y = 10078, grid_size=3):
    grid_points = {}
    sample_idxs = sample_df.keys()
    for sample_idx in set(sample_idxs):
        if max_x is None:
            max_x = sample_df[sample_idx][x_col].max()
        if max_y is None:
            max_y = sample_df[sample_idx][y_col].max()
        x_grid = np.linspace(1000, max_x, grid_size+1)
        y_grid = np.linspace(1000, max_y, grid_size+1)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points[sample_idx] = np.column_stack((xx.ravel(), yy.ravel()))
    return grid_points

def split_data_into_grids(sample_dfs, x_col, y_col, grid_points, grid_size = 3):
    grid_samples = {}
    grid_locs = {}
    sample_idxs = sample_dfs.keys()
    for sample_idx, sample_df in sample_dfs.items():
        grid_point = grid_points[sample_idx]
        grid_samples[sample_idx] = {}
        grid_locs[sample_idx] = {}
        j = 0
        grid_ind = []
        for i in range(grid_size**2):
            grid_locs[sample_idx][i] = pd.DataFrame(columns = [x_col, y_col])
            if (i+j+1) % (grid_size+1) == 0:
                j += 1
            grid_samples[sample_idx][i] = sample_df[(sample_df[x_col] >= grid_point[i+j][0]) & 
                                        (sample_df[y_col] >= grid_point[i+j][1]) &
                                        (sample_df[x_col] < grid_point[i+j+grid_size+2][0]) &
                                        (sample_df[y_col] < grid_point[i+j+grid_size+2][1])]
            grid_locs[sample_idx][i][x_col] = np.maximum(grid_samples[sample_idx][i][x_col]-grid_point[i+j][0],0)
            grid_locs[sample_idx][i][y_col] = np.maximum(grid_samples[sample_idx][i][y_col]-grid_point[i+j][1],0)
            grid_ind.append([i,j])
    return grid_samples, grid_locs, grid_ind

def shuffle_grids(grid_points, grid_samples, grid_locs, grid_ind, x_col, y_col, grid_size = 3, inside_only = False, seed=2024):
    np.random.seed(seed)
    shuffled_grid_samples = {}
    for sample_idx, grid_sample in grid_samples.items():
        temp_df = pd.DataFrame(columns = grid_sample[0].columns)
        if inside_only is True:
            insides = map_for_shuffle_inside(grid_size = grid_size)
            indices = np.array(list(range(grid_size**2)))
            temp_indices = np.random.choice(insides, size = (grid_size-2)**2, replace = True)
            indices[insides] = temp_indices
        else:
            indices = np.random.choice(list(grid_sample.keys()), size=len(list(grid_sample.keys())), replace=True)
        for grid_idx, grid_df in grid_sample.items():
            shuffled_df = grid_sample[indices[grid_idx]]
            # shuffled_df.index = grid_df.index
            shuffled_df.loc[:,x_col] = grid_locs[sample_idx][indices[grid_idx]][x_col]+grid_points[sample_idx][np.sum(grid_ind[grid_idx])][0]
            shuffled_df.loc[:,y_col] = grid_locs[sample_idx][indices[grid_idx]][y_col]+grid_points[sample_idx][np.sum(grid_ind[grid_idx])][1]
            temp_df = pd.concat([temp_df,shuffled_df], ignore_index = True)
        temp_df.reset_index()
        shuffled_grid_samples[sample_idx] = temp_df
        shuffled_grid_samples[sample_idx].isb = shuffled_grid_samples[sample_idx].isb.astype(bool) # for plotting
    return shuffled_grid_samples

def shuffle_points_within_grids(grid_samples, x_col, y_col, seed = 2024):
    np.random.seed(seed)
    shuffled_grid_samples = {}
    for sample_idx, grid_sample in grid_samples.items():
        temp_df = pd.DataFrame(columns = grid_sample[0].columns)
        for grid_idx, grid_df in grid_sample.items():
            indices = np.random.choice(grid_df.index, size=len(grid_df), replace=True)
            shuffled_df = grid_df.loc[indices]
            # shuffled_df.index = grid_df.index
            shuffled_df[x_col] = grid_df[x_col]
            shuffled_df[y_col] = grid_df[y_col]
            temp_df = pd.concat([temp_df,shuffled_df], ignore_index = True)
        temp_df.reset_index()
        shuffled_grid_samples[sample_idx] = temp_df
        shuffled_grid_samples[sample_idx].isb = shuffled_grid_samples[sample_idx].isb.astype(bool) # for plotting
    return shuffled_grid_samples

def make_suffled_df(df, x_col, y_col, grid_size= 3, inside_only = False, seed = 1234):
    a = create_grid(df,x_col, y_col, grid_size = grid_size)
    b, c, d = split_data_into_grids(df, x_col, y_col, a, grid_size = grid_size)
    c = shuffle_grids(a,b,c,d, x_col, y_col, seed = seed, grid_size = grid_size, inside_only = inside_only)
    return c

def map_for_shuffle_inside(grid_size = 4):
    start_num = grid_size + 1
    initial_ran = list(range(start_num, start_num+grid_size-2))
    output_list = []
    for i in np.arange(grid_size-3+1):
        add= [x+grid_size*i for x in initial_ran]
        output_list.extend(add)
    return np.array(output_list)

def generate_block_bootstrap_graphs(df, marker_list, x_coord, y_coord, grid_size, n_samples = 20, 
                                    radius_r = 100, seed = 1):
    samples = {}
    sub_samples = {}
    for i in np.arange(n_samples):
        shuffle_data = make_suffled_df(df, x_coord, y_coord, grid_size, seed = seed + i)
        G, graph_list, _,_ = convert_to_graph(shuffle_data, x_coord, y_coord, z_col=None,
                     n_neighbours = 0, features='markers', arcsinh = True, processing ='znorm', 
                     marker_list = marker_list,
                     radius_knn = radius_r, bw = None)
        samples[i] = G
        sub_samples[i] = graph_list
    return samples, sub_samples
        