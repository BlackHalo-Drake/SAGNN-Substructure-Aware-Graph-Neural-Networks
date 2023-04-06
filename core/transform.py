
from torch_geometric.data import Data
from core.transform_utils.sampling import *
from core.transform_utils.subgraph_extractors import *
import dgl
import re
import itertools
from core.PE_init import lap_positional_encoding, init_positional_encoding
import networkx as nx
from core.config import cfg

import matplotlib.pyplot as plt
class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        self.g = None
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')]+'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            # should use number of subgraphs or number of supernodes.
            return 1+getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            # batched_edge_attr[subgraphs_edges_mapper] shoud be batched_combined_subgraphs_edge_attr
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

class SubgraphsTransform(object):
    def __init__(self, cfg):
        super().__init__()
        self.num_hops = cfg.subgraph.hops
        self.cut_times = cfg.model.cut_times
        self.fullgraph_pos_enc_dim = cfg.model.fullgraph_pos_enc_dim
        self.egograph_pos_enc_dim = cfg.model.egograph_pos_enc_dim
        self.cutgraph_pos_enc_dim = cfg.model.cutgraph_pos_enc_dim
        self.embedding_type = cfg.model.embedding_type


    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        graph = dgl.DGLGraph((data.edge_index[0], data.edge_index[1]))
        if graph.num_nodes() < data.num_nodes:
            offset = data.num_nodes - graph.num_nodes()
            for i in range(offset):
                graph.add_nodes(1)
        # Step 2: extract subgraphs
        subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = extract_subgraphs(data.edge_index, data.num_nodes, self.num_hops)
        subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense)

        data.subgraphs_batch = subgraphs_nodes[0]
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        Ego_RWPE = []
        if self.embedding_type == 'lap_pe':
            global_rwpe = lap_positional_encoding(graph, self.fullgraph_pos_enc_dim)
        elif self.embedding_type == 'rand_walk':
            global_rwpe = init_positional_encoding(graph, self.fullgraph_pos_enc_dim)

        for i in range(subgraphs_edges_mask.shape[0]):
            mask = subgraphs_nodes_mask[i]
            nodes = graph.nodes()
            target_nodes = nodes[mask]
            sub_g = dgl.node_subgraph(graph, target_nodes)
            if self.embedding_type == 'lap_pe':
                ego_rwpe = lap_positional_encoding(sub_g, self.egograph_pos_enc_dim)
            elif self.embedding_type == 'rand_walk':
                ego_rwpe = init_positional_encoding(sub_g, self.egograph_pos_enc_dim)
            else:
                ego_rwpe = init_positional_encoding(sub_g, self.egograph_pos_enc_dim)
            Ego_RWPE.append(ego_rwpe)
        Ego_RWPE = torch.cat(Ego_RWPE, 0)
        graph_nx = dgl.to_networkx(graph)
        graph_nx_undirected = graph_nx.to_undirected()
        target_g_list = []
        comp = nx.algorithms.community.girvan_newman(graph_nx_undirected)
        connected_components = list(nx.algorithms.connected_components(graph_nx_undirected))
        if len(connected_components) >= self.cut_times:
            for item in connected_components:
                target_g_list.append(graph_nx.subgraph(item))
        else:
            ggg = None
            limited = itertools.takewhile(lambda c: len(c) <= self.cut_times, comp)
            for communities in limited:
                ggg = (tuple(sorted(c) for c in communities))
            for i in ggg:
                target_g_list.append(graph_nx.subgraph(i))
        Cut_RWPE = []
        subgraph_x_index = []

        for g in target_g_list:
            subgraph_x_index.append(torch.tensor(list(g.nodes)))
            g_dgl = dgl.from_networkx(g)
            if self.embedding_type == 'lap_pe':
                cut_rwpe = lap_positional_encoding(g_dgl, self.cutgraph_pos_enc_dim)
            elif self.embedding_type == 'rand_walk':
                cut_rwpe = init_positional_encoding(g_dgl, self.cutgraph_pos_enc_dim)
            else:
                cut_rwpe = init_positional_encoding(g_dgl, self.cutgraph_pos_enc_dim)
            Cut_RWPE.append(cut_rwpe)

        data.hop_indicator = hop_indicator
        data.cut_RWPE = torch.cat(Cut_RWPE, dim=0) #cut嵌入
        data.ego_RWPE = Ego_RWPE  #hop嵌入
        data.glo_RWPE = global_rwpe  #全局嵌入
        data.subgraph_x_index = torch.cat(subgraph_x_index, dim=-1)
        data.__num_nodes__ = data.num_nodes # set number of nodes of the current graph
        return data

"""
    Helpers
"""
import numpy as np

def select_subgraphs(subgraphs_nodes,selected_subgraphs):
    selected_subgraphs = np.sort(selected_subgraphs)

    selected_nodes_mask = check_values_in_set(subgraphs_nodes[0], selected_subgraphs)
    nodes_batch = subgraphs_nodes[0][selected_nodes_mask]
    batch_mapper = torch.zeros(1 + nodes_batch.max(), dtype=torch.long)
    batch_mapper[selected_subgraphs] = torch.arange(len(selected_subgraphs))

    selected_subgraphs_nodes = batch_mapper[nodes_batch], subgraphs_nodes[1][selected_nodes_mask]
    return selected_subgraphs_nodes

def to_sparse(node_mask, edge_mask, hop_indicator):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
    return subgraphs_nodes, subgraphs_edges, hop_indicator

def extract_subgraphs(edge_index, num_nodes, num_hops, sparse=False):
    node_mask, hop_indicator = k_hop_subgraph(edge_index, num_nodes, num_hops)
    edge_mask = node_mask[:, edge_index[0]] & node_mask[:, edge_index[1]] # N x E dense mask matrix
    if not sparse:
        return node_mask, edge_mask, hop_indicator
    else:
        return to_sparse(node_mask, edge_mask, hop_indicator)

def subsampling_subgraphs(edge_index, subgraphs_nodes, num_nodes=None, sampling_mode='shortest_path', random_init=False, minimum_redundancy=0,
                          shortest_path_mode_stride=2, random_mode_sampling_rate=0.5):

    assert sampling_mode in ['shortest_path', 'random', 'min_set_cover']
    if sampling_mode == 'random':
        selected_subgraphs, node_selected_times = random_sampling(subgraphs_nodes, num_nodes=num_nodes, rate=random_mode_sampling_rate, minimum_redundancy=minimum_redundancy)
    if sampling_mode == 'shortest_path':
        selected_subgraphs, node_selected_times = shortest_path_sampling(edge_index, subgraphs_nodes, num_nodes=num_nodes, minimum_redundancy=minimum_redundancy,
                                                                         stride=max(1, shortest_path_mode_stride), random_init=random_init)
    if sampling_mode in ['min_set_cover']:
        assert subgraphs_nodes.size(0) == num_nodes # make sure this is subgraph_nodes_masks
        selected_subgraphs, node_selected_times = min_set_cover_sampling(edge_index, subgraphs_nodes,
                                                                         minimum_redundancy=minimum_redundancy, random_init=random_init)
    return selected_subgraphs, node_selected_times

def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected)*num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs


def hops_to_selected_nodes(edge_index, selected_nodes, num_nodes=None):
    row, col = edge_index
    if num_nodes is None:
        num_nodes = 1 + edge_index.max()
    hop_indicator = row.new_full((num_nodes,), -1)
    bipartitie_indicator = row.new_full(row.shape, -1)
    hop_indicator[selected_nodes] = 0
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    selected_nodes = (hop_indicator == 0)
    i = 1
    while hop_indicator.min() < 0:
        source_near_edges = selected_nodes[row]
        node_mask.fill_(False)
        node_mask[col[source_near_edges]] = True
        selected_nodes = (hop_indicator==-1) & node_mask
        bipartitie_between_source_target = source_near_edges & selected_nodes[col]
        bipartitie_indicator[bipartitie_between_source_target] = i
        hop_indicator[selected_nodes] = i
        i += 1

    return hop_indicator, bipartitie_indicator