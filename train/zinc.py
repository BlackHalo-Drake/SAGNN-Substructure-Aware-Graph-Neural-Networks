
import torch
from core.config import cfg, update_cfg
from core.train_helper import run
from core.model import SAGNN
from core.transform import SubgraphsTransform

from torch_geometric.datasets import ZINC
from core.utils import calculate_stats
import os.path as osp
import os
def create_dataset(cfg):
    # No need to do offline transformation
    cfg.processed_dir = './data/ZINC/subset/processed/processed'
    transform = SubgraphsTransform(cfg)

    root = 'data/ZINC'
    train_dataset = ZINC(root, subset=True, split='train', transform=transform)
    val_dataset = ZINC(root, subset=True, split='val', transform=transform)
    test_dataset = ZINC(root, subset=True, split='test', transform=transform)

    # When without randomness, transform the data to save a bit time
    if os.path.exists(cfg.processed_dir):
        train_dataset = torch.load(osp.join(cfg.processed_dir, 'train_processed.pt'))
        val_dataset = torch.load(osp.join(cfg.processed_dir, 'val_processed.pt'))
        test_dataset = torch.load(osp.join(cfg.processed_dir, 'test_processed.pt'))
    else:
        os.makedirs(cfg.processed_dir)
        train_dataset = [x for x in train_dataset]
        val_dataset = [x for x in val_dataset]
        test_dataset = [x for x in test_dataset]
        torch.save(train_dataset, osp.join(cfg.processed_dir, 'train_processed.pt'))
        torch.save(val_dataset, osp.join(cfg.processed_dir, 'val_processed.pt'))
        torch.save(test_dataset, osp.join(cfg.processed_dir, 'test_processed.pt'))

    print('------------Train--------------')
    calculate_stats(train_dataset)
    print('------------Validation--------------')
    calculate_stats(val_dataset)
    print('------------Test--------------')
    calculate_stats(test_dataset)
    print('------------------------------')

    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = SAGNN(None, None,
                        nhid=cfg.model.hidden_size,
                        nout=1,
                        nlayer_outer=cfg.model.num_layers,
                        nlayer_inner=cfg.model.mini_layers,
                        gnn_types=[cfg.model.gnn_type],
                        embedding_types=cfg.model.embedding_type,
                        hop_dim=cfg.model.hops_dim,
                        fullgraph_pos_enc_dim=cfg.model.fullgraph_pos_enc_dim,
                        egograph_pos_enc_dim=cfg.model.egograph_pos_enc_dim,
                        cutgraph_pos_enc_dim=cfg.model.cutgraph_pos_enc_dim,
                        pos_enc_dim=cfg.model.pos_enc_dim,
                        embedding_learnable=cfg.model.embedding_learnable,
                        pooling=cfg.model.pool,
                        global_embedding=cfg.model.global_embedding,
                        dropout=cfg.train.dropout
)
    return model
def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        loss = (model(data).squeeze() - y).abs().mean()
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    total_error = 0
    N = 0
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        total_error += (model(data).squeeze() - y).abs().sum().item()
        N += num_graphs
    test_perf = - total_error / N
    return test_perf


if __name__ == '__main__':
    # get config
    cfg.merge_from_file('train/configs/zinc.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)