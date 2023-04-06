import numpy as np
def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')

