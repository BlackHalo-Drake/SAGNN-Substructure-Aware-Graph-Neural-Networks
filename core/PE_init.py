import dgl
import torch
from scipy import sparse as sp
import numpy as np

def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N  # 得到拉普拉斯矩阵

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())  # 获得特征值、特征向量
    idx = EigVal.argsort()  # increasing order #从小到大排列特征值
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])  # 让特征值和特征向量按照特征值从小到大顺序一一对应

    pos_enc_emb = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    if pos_enc_emb.shape[-1] < pos_enc_dim:
        offset = pos_enc_dim - pos_enc_emb.shape[-1]
        pos_enc_emb = torch.cat((pos_enc_emb, torch.zeros(pos_enc_emb.shape[0], offset)), dim=-1)


    return pos_enc_emb


def init_positional_encoding(g, pos_enc_dim, type_init='rand_walk'):
    """
        Initializing positional encoding with RWPE
    """

    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
        RW = A * Dinv
        M = RW  # 随机游走一次后的子图

        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc - 1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE, dim=-1)
        g.ndata['pos_enc'] = PE

    return PE