import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import core.model_utils.pyg_gnn_wrapper as gnn_wrapper
from core.model_utils.elements import MLP, DiscreteEncoder, Identity, VNUpdate
BN = True

class SAGNN(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer_outer, nlayer_inner, gnn_types, embedding_types,
                 dropout=0,
                 hop_dim=0,
                 fullgraph_pos_enc_dim=16,
                 egograph_pos_enc_dim=8,
                 cutgraph_pos_enc_dim=8,
                 pos_enc_dim=16,
                 node_embedding=True,
                 embedding_learnable=0,
                 bn=BN,
                 vn=False,
                 res=True,
                 pooling='mean',
                 global_embedding=True):
        super().__init__()
        # nfeat_in is None: discrete input features
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        self.with_global_emb = global_embedding
        self.vn = vn
        self.vn_aggregators = nn.ModuleList([VNUpdate(nhid) for _ in range(nlayer_outer)])
        self.vn_aggregators2 = nn.ModuleList([VNUpdate(nhid) for _ in range(nlayer_outer)])
        # layers
        edge_emd_dim = nhid
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(edge_emd_dim) if nfeat_edge is None else MLP(nfeat_edge, edge_emd_dim, 1)
                                            for _ in range(nlayer_outer)])

        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer_outer)])
        self.norms1 = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer_outer)])
        self.dropout = dropout
        self.gnn_type = gnn_types[0]

        self.num_inner_layers = nlayer_inner
        self.node_embedding = node_embedding
        self.embedding_learnable = embedding_learnable
        self.hop_dim = hop_dim
        self.res = res
        self.pooling = pooling

        self.output_decoder = nn.Sequential(
            nn.Linear(nhid * 2, nhid),  #
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(nhid, nout),
        )

        self.output_decoder2 = MLP(nhid, nout, nlayer=2, with_final_activation=False)


        self.traditional_gnns = nn.ModuleList([getattr(gnn_wrapper, gnn_types[0])(nhid, nhid, bias=not bn) for _ in range(nlayer_outer)])
        self.subgraph_gnns = nn.ModuleList([getattr(gnn_wrapper, gnn_types[0])(nhid, nhid, bias=not bn) for _ in range(nlayer_outer)])

        self.embedding_type = embedding_types
        self.fullgraph_pos_enc_dim = fullgraph_pos_enc_dim
        self.egograph_pos_enc_dim = egograph_pos_enc_dim
        self.cutgraph_pos_enc_dim = cutgraph_pos_enc_dim
        self.pos_enc_dim = pos_enc_dim
        if self.embedding_type is not None:
            self.ego_sub_emb = nn.Linear(self.egograph_pos_enc_dim, nhid)
            self.global_emb = nn.Linear(self.fullgraph_pos_enc_dim, nhid)
            self.cut_sub_emb= nn.Linear(self.cutgraph_pos_enc_dim, nhid)
            self.global_out = nn.Linear(nhid, self.pos_enc_dim)
            self.ego_sub_out = nn.Linear(nhid, self.pos_enc_dim)
            self.cut_sub_out = nn.Linear(nhid, self.pos_enc_dim)
            if self.with_global_emb:
                self.Whp = nn.Linear(nhid+self.pos_enc_dim*2, nhid)
                self.Whp2 = nn.Linear(nhid+self.pos_enc_dim*2, nhid)
            else:
                self.Whp = nn.Linear(nhid+self.pos_enc_dim, nhid)
                self.Whp2 = nn.Linear(nhid+self.pos_enc_dim, nhid)
        if self.with_global_emb:
            self.merge = nn.Linear(3 * nhid, nhid)
            self.merge2 = nn.Linear(3 * nhid, nhid)
        else:
            self.merge = nn.Linear(2 * nhid, nhid)
            self.merge2 = nn.Linear(2 * nhid, nhid)


    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.merge.reset_parameters()
        if self.embedding_type is not None:
            self.global_emb.reset_parameters()
            self.ego_sub_emb.reset_parameters()
            self.cut_sub_emb.reset_parameters()
            self.global_out.reset_parameters()
            self.ego_sub_out.reset_parameters()
            self.cut_sub_out.reset_parameters()
            self.Whp.reset_parameters()
            self.Whp2.reset_parameters()
        self.merge2.reset_parameters()
        self.output_decoder2.reset_parameters()
        self.output_decoder[0].reset_parameters()
        self.output_decoder[-1].reset_parameters()
        for edge_encoder, norm, norm1, old, sub_gnn, vn, vn2 in zip(self.edge_encoders, self.norms, self.norms1, self.traditional_gnns, self.subgraph_gnns,  self.vn_aggregators, self.vn_aggregators2):
            edge_encoder.reset_parameters()
            norm.reset_parameters()
            norm1.reset_parameters()
            old.reset_parameters()
            sub_gnn.reset_parameters()
            vn.reset_parameters()
            vn2.reset_parameters()

    def forward(self, data):
        ego_sub_embedding = data.ego_RWPE
        global_embedding = data.glo_RWPE
        cut_sub_embedding = data.cut_RWPE
        subgraph_x_index = data.subgraph_x_index
        x = data.x if len(data.x.shape) <= 2 else data.x.squeeze(-1)
        x = self.input_encoder(x)
        ori_edge_attr = data.edge_attr
        x_cut = None

        if ori_edge_attr is None:
            ori_edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))

        if self.embedding_type in ['rand_walk', 'lap_pe']:
            ego_sub_embedding = self.ego_sub_emb(ego_sub_embedding)
            global_embedding = self.global_emb(global_embedding)
            cut_sub_embedding = self.cut_sub_emb(cut_sub_embedding)

        cut_sub_embedding_trans = torch.zeros_like(cut_sub_embedding).to(cut_sub_embedding)
        cut_sub_embedding_trans[subgraph_x_index] = cut_sub_embedding
        ego_sub_embedding_pooled = scatter(ego_sub_embedding, data.subgraphs_batch, dim=0, reduce=self.pooling)
        if self.embedding_type not in ['rand_walk', 'lap_pe']:   # other encoding
            previous_x = x
        else:
            if self.with_global_emb:
                x_ego = self.merge(torch.cat((x, ego_sub_embedding_pooled, global_embedding), dim=-1))
                x_cut = self.merge2(torch.cat((x, cut_sub_embedding_trans, global_embedding), dim=-1))
            else:
                x_ego = self.merge(torch.cat((x, ego_sub_embedding_pooled), dim=-1))
                x_cut = self.merge2(torch.cat((x, cut_sub_embedding_trans), dim=-1))
            previous_x_ego = x_ego
            previous_x_cut = x_cut

        if self.vn:
            virtual_node1 = None
            virtual_node = None

        for i, (edge_encoder, ego_sub_gnn, norm, norm1, cut_sub_gnn, vn_aggregator, vn_aggregator2) in enumerate(zip(self.edge_encoders, \
                                                                                                                     self.traditional_gnns, self.norms, self.norms1,  self.subgraph_gnns, self.vn_aggregators, self.vn_aggregators2)):
            # if ori_edge_attr is not None: # Encode edge attr for each layer
            data.edge_attr = edge_encoder(ori_edge_attr) #对边特征进行编码（embedding）
            data.x = x
            if self.embedding_type not in ['rand_walk', 'lap_pe']:
                # standard message passing nn #标准的GNN传递
                x = ego_sub_gnn(data.x, data.edge_index, data.edge_attr)
                x = F.dropout(F.relu(norm(x)), self.dropout, training=self.training)
                if self.res:
                    x = x.clone() + previous_x
                    previous_x = x # for residual connection
                if self.vn:
                    virtual_node, x = vn_aggregator2(virtual_node, x, data.batch)

            else:
                x_ego = ego_sub_gnn(x_ego, data.edge_index, data.edge_attr) #
                x_cut = cut_sub_gnn(x_cut, data.edge_index, data.edge_attr)
                x_ego = F.dropout(F.relu(norm(x_ego)), self.dropout, training=self.training)
                x_cut = F.dropout(F.relu(norm1(x_cut)), self.dropout, training=self.training)
                if self.res:
                    x_ego = x_ego.clone() + previous_x_ego
                    previous_x_ego = x_ego # for residual connection
                    x_cut = x_cut.clone() + previous_x_cut
                    previous_x_cut = x_cut
                if self.vn:
                    virtual_node1, x_ego = vn_aggregator(virtual_node1, x_ego, data.batch)
                    virtual_node, x_cut = vn_aggregator2(virtual_node, x_cut, data.batch)

        hp = None
        hp2 = None
        glo_emb = None

        if self.embedding_type in ['rand_walk', 'lap_pe']:
            ego_sub_emb = torch.relu(self.ego_sub_out(ego_sub_embedding_pooled))
            cut_sub_emb = torch.relu(self.cut_sub_out(cut_sub_embedding_trans))
            if self.with_global_emb:
                glo_emb = torch.relu(self.global_out(global_embedding))
                hp = self.Whp(torch.cat((x_ego, glo_emb, ego_sub_emb), dim=-1))
                hp2 = self.Whp2(torch.cat((x_cut, glo_emb, cut_sub_emb), dim=-1))
            else:
                hp = self.Whp(torch.cat((x_ego, ego_sub_emb), dim=-1))
                hp2 = self.Whp2(torch.cat((x_cut, cut_sub_emb), dim=-1))
        else:
            hp = x

        if hp2 is None:
            res = scatter(hp, data.batch, dim=0, reduce=self.pooling)
            res = F.dropout(res, self.dropout, training=self.training)
            res = self.output_decoder2(res)
        else:
            res = scatter(hp, data.batch, dim=0, reduce=self.pooling)
            res_cut = scatter(hp2, data.batch, dim=0, reduce=self.pooling)
            res = F.dropout(res, self.dropout, training=self.training)
            res_cut = F.dropout(res_cut, self.dropout, training=self.training)
            res = self.output_decoder(torch.cat((res, res_cut), dim=-1))
        return res

