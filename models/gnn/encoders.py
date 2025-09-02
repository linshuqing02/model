import torch.nn
from typing import Any, Dict, List, Tuple, Optional, Union
from ogb.utils.mol import smiles2graph
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder
import numpy as np

def get_fc_layers(hidden_sizes,
        dropout=0, batchnorm=False,
        no_last_dropout=True, no_last_activation=True):
    act_fn = torch.nn.LeakyReLU()
    layers = []
    for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
        layers.append(torch.nn.Linear(in_dim, out_dim))
        if not no_last_activation or i != len(hidden_sizes) - 2:
            layers.append(act_fn)
        if dropout > 0:
            if not no_last_dropout or i != len(hidden_sizes) - 2:
                layers.append(torch.nn.Dropout(dropout))
        if batchnorm and i != len(hidden_sizes) - 2:
            layers.append(torch.nn.BatchNorm1d(out_dim))
    return torch.nn.Sequential(*layers)


def graph_batch_from_smiles(smiles_list, device):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]

    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph["edge_index"] + lstnode)
        edge_feats.append(graph["edge_feat"])
        node_feats.append(graph["node_feat"])
        lstnode += graph["num_nodes"]
        batch.append(np.ones(graph["num_nodes"], dtype=np.int64) * idx)

    result = {
        "edge_index": np.concatenate(edge_idxes, axis=-1),
        "edge_attr": np.concatenate(edge_feats, axis=0),
        "batch": np.concatenate(batch, axis=0),
        "x": np.concatenate(node_feats, axis=0),
    }
    result = {k: torch.from_numpy(v).to(device) for k, v in result.items()}
    result["num_nodes"] = lstnode
    result["num_edges"] = result["edge_index"].shape[1]
    return result


class StaticParaDict(torch.nn.Module):
    def __init__(self, **kwargs):
        super(StaticParaDict, self).__init__()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, torch.nn.Parameter(v, requires_grad=False))
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                setattr(self, k, torch.nn.Parameter(v, requires_grad=False))
            else:
                setattr(self, k, v)

    def forward(self, key: str) -> Any:
        return getattr(self, key)

    def __getitem__(self, key: str) -> Any:
        return self(key)

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            value = torch.nn.Parameter(value, requires_grad=False)
        setattr(self, key, value)


class GINConv(torch.nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super(GINConv, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2 * embedding_dim),
            torch.nn.BatchNorm1d(2 * embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=embedding_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        edge_feats = self.bond_encoder(edge_feats)
        message_node = torch.index_select(input=node_feats, dim=0, index=edge_index[1])
        message = torch.relu(message_node + edge_feats)
        dim = message.shape[-1]

        message_reduce = torch.zeros(num_nodes, dim).to(message)
        index = edge_index[0].unsqueeze(-1).repeat(1, dim)
        message_reduce.scatter_add_(dim=0, index=index, src=message)

        return self.mlp((1 + self.eps) * node_feats + message_reduce)


class GINGraph(torch.nn.Module):
    def __init__(
        self, num_layers: int = 4, embedding_dim: int = 64, dropout: float = 0.7
    ):
        super(GINGraph, self).__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim=embedding_dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout_fun = torch.nn.Dropout(dropout)
        for layer in range(self.num_layers):
            self.convs.append(GINConv(embedding_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(embedding_dim))

    def forward(self, graph: Dict[str, Union[int, torch.Tensor]]) -> torch.Tensor:
        h_list = [self.atom_encoder(graph["x"])]
        for layer in range(self.num_layers):
            h = self.batch_norms[layer](
                self.convs[layer](
                    node_feats=h_list[layer],
                    edge_feats=graph["edge_attr"],
                    edge_index=graph["edge_index"],
                    num_nodes=graph["num_nodes"],
                    num_edges=graph["num_edges"],
                )
            )
            if layer != self.num_layers - 1:
                h = self.dropout_fun(torch.relu(h))
            else:
                h = self.dropout_fun(h)
            h_list.append(h)

        batch_size, dim = graph["batch"].max().item() + 1, h_list[-1].shape[-1]
        out_feat = torch.zeros(batch_size, dim).to(h_list[-1])
        cnt = torch.zeros_like(out_feat).to(out_feat)
        index = graph["batch"].unsqueeze(-1).repeat(1, dim)

        out_feat.scatter_add_(dim=0, index=index, src=h_list[-1])
        cnt.scatter_add_(
            dim=0, index=index, src=torch.ones_like(h_list[-1]).to(h_list[-1])
        )

        return out_feat / (cnt + 1e-9)