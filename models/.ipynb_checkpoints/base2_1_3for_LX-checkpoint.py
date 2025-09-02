import torch.nn as nn
import torch.nn
from typing import Any, Dict, Union
from pyhealth.tokenizer import Tokenizer
from pyhealth.models import BaseModel
from ogb.utils.mol import smiles2graph
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder
import numpy as np
from .gnn import DrugGVPModel
from pyhealth.medcode import ATC
from typing import Any, Dict, List, Tuple, Optional, Union
from torch_geometric import loader
from models.gnn import drug_sdf_db
import dill
from rdkit import Chem
import torch.nn.functional as F


def aggregate_tensors(tensors, device):
    """
    将多个张量聚合到最大长度
    :param tensors: list of tensors
    :return: 聚合后的张量, 每个batch的长度
    """
    max_len = max([x.size(1) for x in tensors])
    padded_inputs = []
    lengths = []

    for x in tensors:
        lengths.append(x.size(0))
        padding = torch.zeros(x.size(0), max_len - x.size(1), x.size(2)).to(device)
        padded_x = torch.cat((x, padding), dim=1)
        padded_inputs.append(padded_x)

    aggregated_tensor = torch.cat(padded_inputs, dim=0)
    return aggregated_tensor, lengths


def split_tensor(tensor, lengths, max_len):
    """
    将聚合的张量拆分为原始形状
    :param tensor: 聚合的张量
    :param lengths: 每个batch的长度
    :param max_len: 最大长度
    :return: 拆分后的张量列表
    """
    index = 0
    outputs = []

    for length in lengths:
        output_tensor = tensor[index:index + length]
        outputs.append(output_tensor)
        index += length

    outputs = [x[:, :max_len, :] for x in outputs]
    return outputs


def extract_and_transpose(tensor_list):
    """
    提取每个张量的最后一个序列并转置
    :param tensor_list: list of tensors
    :return: 处理后的张量列表
    """
    processed_tensors = []
    for tensor in tensor_list:
        last_seq = tensor[:, -1:, :]  # 提取最后一个序列
        transposed_seq = last_seq.transpose(0, 1)  # 转置
        processed_tensors.append(transposed_seq)
    return processed_tensors


def graph_batch_from_smiles(smiles_list, device):
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs_valid = []
    # 预处理：确保输入是列表且不为空
    if not isinstance(smiles_list, list):
        smiles_list = [smiles_list] if smiles_list else []
    
    # 安全处理每个SMILES
    for smiles in smiles_list:
        try:
            # 检查基本有效性
            if not isinstance(smiles, str) or not smiles.strip():
                continue
                
            # RDKit分子转换
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                continue
                
            # 尝试生成图结构
            graph = smiles2graph(smiles)
            if not isinstance(graph, dict):
                continue
                
            # 检查必要字段是否存在
            required_keys = {"edge_index", "edge_feat", "node_feat", "num_nodes"}
            if not all(key in graph for key in required_keys):
                continue
                
            graphs_valid.append(graph)
            
        except Exception as e:
            continue

    for idx, graph in enumerate(graphs_valid):
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


def get_smiles_list(drug_list):
    """Generates the list of SMILES strings."""
    atc3_to_smiles = {}
    atc = ATC()
    for code in atc.graph.nodes:
        if len(code) != 7:
            continue
        code_atc3 = ATC.convert(code, level=3)
        smiles = atc.graph.nodes[code]["smiles"]
        if smiles != smiles:
            continue
        atc3_to_smiles[code_atc3] = atc3_to_smiles.get(code_atc3, []) + [smiles]
    atc3_to_smiles = {k: v[:1] for k, v in atc3_to_smiles.items()}  #230种药物的smiles
    smiles_list = []
    for drug in drug_list:
        if drug in atc3_to_smiles:
            smiles_list.append(atc3_to_smiles[drug][0])
        # else:
        #     smiles_list.append('[Na+].[Cl-]')
    return smiles_list 


def contrastive_loss(anchor, positive, temperature=0.07):
    # anchor: (N, D)
    # positive: (N, D)
    anchor_norm = F.normalize(anchor, dim=1)  # (N, D)
    positive_norm = F.normalize(positive, dim=1)  # (N, D)
    logits = torch.matmul(anchor_norm, positive_norm.t())  # (N, N)
    logits = logits / temperature
    labels = torch.arange(anchor.size(0)).long().to(anchor.device)  # (N,)
    loss = F.cross_entropy(logits, labels)
    return loss


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
    
class Base2_1_3(BaseModel):
    def __init__(
            self,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            task_dataset,
            args,
            drug_geo_loader,
            average_projection,
            all_smiles_flatten,
            embedding_dim=128,
            dropout=0.7,
            GNN_layers=4
    ):
        super(Base2_1_3, self).__init__(
            dataset=task_dataset,
            feature_keys=["conditions", "procedures", "drugs_hist"],
            label_key="drugs",
        )

        self.drug_node_in_dim=[66, 1]
        self.drug_node_h_dims=[128, 64]
        self.drug_edge_in_dim=[16, 1]
        self.drug_edge_h_dims=[32, 1]
        self.drug_fc_dims=[1024, 128]
        self.drug_emb_dim = self.drug_node_h_dims[0]

        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event
        self.monitor_event_token = Tokenizers_monitor_event
        self.drug_geo_loader = drug_geo_loader
        self.all_smiles_flatten = all_smiles_flatten
        self.label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('drugs'))
        self.label_size = output_size

        self.drug_to_index = {drug: idx for idx, drug in enumerate(task_dataset.get_all_tokens('drugs'))}

        self.average_projection = torch.nn.Parameter(
            average_projection, requires_grad=False
        ) # 195,528
        self.molecule_graphs = StaticParaDict(
            **graph_batch_from_smiles(self.all_smiles_flatten, device)
        )

        GNN_para = {
            "num_layers": GNN_layers,
            "dropout": dropout,
            "embedding_dim": embedding_dim,
        }
        
        # 2d GNN
        self.molecule_encoder = GINGraph(**GNN_para)
        # 3d GVP
        self.drug_model = DrugGVPModel(
            node_in_dim=self.drug_node_in_dim, node_h_dim=self.drug_node_h_dims,
            edge_in_dim=self.drug_edge_in_dim, edge_h_dim=self.drug_edge_h_dims,
        )

        self.smile_attention = nn.Sequential(
            nn.Linear(embedding_dim, 1),  
            nn.Softmax(dim=0)             
        ).to(device)

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # 256 -> 128
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),  # 128 -> 128
            nn.Dropout(dropout)
        ).to(device)

        smiles2cid_path = f'data/{args.dataset}/SMILES2CID.pkl'
        smiles_3d_path = f'data/{args.dataset}/smiles_3d'
        with open(smiles2cid_path, 'rb') as Fin:
            self.smiles2cid = dill.load(Fin)
        self.drug_sdf_dict = drug_sdf_db(smiles_3d_path)

        self.drug_fc = self.get_fc_layers(
            [self.drug_emb_dim] + self.drug_fc_dims,
            dropout=0.25, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        score_extractor = [
            torch.nn.Linear(embedding_dim, embedding_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim // 2, 1),
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        item_num = 3
        self.drug_query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * item_num, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, output_size)
        )

        self.feature_visit_evnet_keys = Tokenizers_visit_event.keys()
        # self.feature_visit_evnet_keys=['conditions', 'procedures']
        self.feature_monitor_event_keys = Tokenizers_monitor_event.keys()
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.device = device

        self.embeddings = nn.ModuleDict()
        # 为每种event（包含monitor和visit）添加一种嵌入
        for feature_key in self.feature_visit_evnet_keys:
            tokenizer = self.visit_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        for feature_key in self.feature_monitor_event_keys:
            tokenizer = self.monitor_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        self.visit_gru = nn.ModuleDict()
        # 为每种visit_event添加一种gru
        for feature_key in self.feature_visit_evnet_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in self.feature_monitor_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in ['weight', 'age']:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.monitor_gru = nn.ModuleDict()
        # 为每种monitor_event添加一种gru
        for feature_key in self.feature_monitor_event_keys:
            self.monitor_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.decoder_2d = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2),  # 128 -> 256
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim), # 256 -> 128
        ).to(device)

        self.decoder_3d = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2),  # 128 -> 256
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim), # 256 -> 128
        ).to(device)


        self.fc_age = nn.Linear(1, self.embedding_dim)
        self.fc_weight = nn.Linear(1, self.embedding_dim)


    def get_fc_layers(self, hidden_sizes,
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


    def forward(self, batch_data):

        batch_size = len(batch_data['visit_id'])
        patient_emb_list = []

        # 2D分子
        molecule_graphs = self.molecule_graphs
        embeddings_2D = self.molecule_encoder(molecule_graphs)  #195,128
        embeddings_2D = torch.mm(self.average_projection, embeddings_2D)  #195,128

        # 3D分子
        embeddings_3D = self.drug_model(next(iter(self.drug_geo_loader)))
        embeddings_3D = self.drug_fc(embeddings_3D) # [528,128]
        embeddings_3D = torch.mm(self.average_projection, embeddings_3D) #(195,128) 

        # fusion
        molecule_embeddings = self.fusion(torch.cat([embeddings_2D, embeddings_3D], dim=-1))  # (195, 128)
        
        # """处理lab, inj"""
        # feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
        # # 迭代处理每一对
        # for feature_key1, feature_key2 in feature_paris:
        #     monitor_emb_list = []
        #     # 先聚合monitor层面，生成batch_size个病人的多次就诊的表征，batch_size * (1, visit, embedding)
        #     for patient in range(batch_size):
        #         x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
        #             batch_data[feature_key1][patient], max_length=(400, 1024)
        #         )
        #         x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
        #         x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
        #             batch_data[feature_key2][patient], max_length=(400, 1024)
        #         )
        #         x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
        #         x1 = self.dropout(self.embeddings[feature_key1](x1))
        #         x2 = self.dropout(self.embeddings[feature_key2](x2))
        #         x = torch.mul(x1, x2)
        #         x = torch.sum(x, dim=2)
        #         monitor_emb_list.append(x)
        #     # 聚合多次的monitor
        #     aggregated_monitor_tensor, lengths = aggregate_tensors(monitor_emb_list, self.device)
        #     output, hidden = self.monitor_gru[feature_key1](aggregated_monitor_tensor)
        #     # 拆分gru的输出
        #     max_len = max([x.size(1) for x in monitor_emb_list])
        #     split_outputs = split_tensor(output, lengths, max_len)
        #     # 提取最后一个序列并转置
        #     visit_emb_list = extract_and_transpose(split_outputs)
        #     # 开始搞visit层面的
        #     aggregated_visit_tensor, lengths = aggregate_tensors(visit_emb_list, self.device)
        #     output, hidden = self.visit_gru[feature_key1](aggregated_visit_tensor)
        #     patient_emb_list.append(hidden.squeeze(dim=0))
        #     # (patient, event)
        
        # """处理weight, age, gender(gender不用加入gru)"""
        # for feature_key in ['weight', 'age']:
        #     x = batch_data[feature_key]
        #     max_length = max(len(sublist) for sublist in x)
        #     # 将每个子列表的元素转换为浮点数，并使用0对齐长度
        #     x = [[float(item) for item in sublist] + [0] * (max_length - len(sublist)) for sublist in x]
        #     x = torch.tensor(x, dtype=torch.float, device=self.device)
        #     num_patients, num_visits = x.shape
        #     x = x.view(-1, 1)  # 变成 (patient * visit, 1)
        #     # 创建一个掩码用于标记输入为0的位置
        #     mask = (x == 0)
        #     if feature_key == 'weight':
        #         x = self.dropout(self.fc_weight(x))
        #     elif feature_key == 'age':
        #         x = self.dropout(self.fc_age(x))
        #     # 对输入为0的位置输出也设为0
        #     x = x * (~mask)
        #     x = x.view(num_patients, num_visits, -1)
        #     output, hidden = self.visit_gru[feature_key](x)
        #     patient_emb_list.append(hidden.squeeze(dim=0))


        """处理cond, proc, drug"""
        for feature_key in self.feature_visit_evnet_keys:
            if feature_key != 'drugs_hist':
                x = self.visit_event_token[feature_key].batch_encode_3d(
                    batch_data[feature_key]
                )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event)
                x = self.dropout(self.embeddings[feature_key](x)) 
                # (patient, visit, event, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit, embedding_dim)

            elif feature_key == 'drugs_hist':
                # x 就是所有visit中的drug的fusion嵌入 (patient, visit, embedding_dim)
                drugs_hist = batch_data['drugs_hist']
                drug_emb_list = [[] for _ in range(batch_size)]  # 初始化每个病人的药物嵌入列表
                drug_emb_2d_origin_list = []
                drug_emb_3d_origin_list = []
                drug_emb_2d_recon_list = []
                drug_emb_3d_recon_list = []

                for i in range(batch_size):
                    if len(drugs_hist[i]) < 2:  # 如果没有药物历史，跳过
                        drug_emb_list[i].append(torch.zeros(1, self.embedding_dim, device=self.device))
                        drug_emb_2d_origin_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                        drug_emb_3d_origin_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                        drug_emb_2d_recon_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                        drug_emb_3d_recon_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                        continue
                    for j in range(len(drugs_hist[i])-1):  #处理每个visit
                        drug_hist = drugs_hist[i][j]
                        if len(drug_hist) == 0:
                            drug_emb_list[i].append(torch.zeros(1, self.embedding_dim, device=self.device))
                            drug_emb_2d_origin_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                            drug_emb_3d_origin_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                            drug_emb_2d_recon_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                            drug_emb_3d_recon_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
                            continue

                        # drug_smiles = get_smiles_list(drug_hist)  # 获取药物的SMILES列表
                        # if not drug_smiles:  # 如果SMILES列表为空，跳过或填充
                        #     drug_emb_list[i].append(torch.zeros(1, self.embedding_dim, device=self.device))
                        #     continue
                        # molecule_graph = StaticParaDict(
                        #     **graph_batch_from_smiles(drug_smiles, self.device)
                        # )   

                        # # 2d
                        # drug_emb_2d = self.molecule_encoder(molecule_graph)
                        # attention_scores_2d = self.smile_attention(drug_emb_2d)  # (num_drugs, 1)
                        # drug_emb_2d = (attention_scores_2d * drug_emb_2d).sum(dim=0).unsqueeze(0) # 1,128

                        # # 3d
                        # druggeolist = [self.drug_sdf_dict[self.smiles2cid[x]].to(self.device) for x in drug_smiles]
                        # if not druggeolist:  # 如果无3D数据，填充零
                        #     drug_emb_3d = torch.zeros(1, self.embedding_dim, device=self.device)
                        # else:
                        #     druggeoloader = loader.DataLoader(dataset=druggeolist, batch_size=len(druggeolist), shuffle=False)
                        #     drug_emb_3d = self.drug_model(next(iter(druggeoloader)))  # (num_drugs, drug_emb_dim)
                        #     drug_emb_3d = self.drug_fc(drug_emb_3d)     # (num_drugs, 128)
                        #     attention_scores_3d = self.smile_attention(drug_emb_3d)  # (num_drugs, 1)
                        #     drug_emb_3d = (attention_scores_3d * drug_emb_3d).sum(dim=0).unsqueeze(0)

                        # 直接从预计算的表征中提取
                        valid_drugs = []
                        drug_indices = []
                        for drug in drug_hist:
                            if drug in self.drug_to_index:
                                valid_drugs.append(drug)
                                drug_indices.append(self.drug_to_index[drug])

                        if not valid_drugs:  # 如果没有有效药物，填充零
                            drug_emb_2d = torch.zeros(1, self.embedding_dim, device=self.device)
                            drug_emb_3d = torch.zeros(1, self.embedding_dim, device=self.device)
                        else:
                            # 直接从预计算的表征中提取
                            drug_emb_2d = embeddings_2D[drug_indices]  # [num_valid_drugs, 128]
                            drug_emb_3d = embeddings_3D[drug_indices]  # [num_valid_drugs, 128]

                            # 注意力聚合
                            attention_scores_2d = self.smile_attention(drug_emb_2d)  # [num_valid_drugs, 1]
                            drug_emb_2d = (attention_scores_2d * drug_emb_2d).sum(dim=0).unsqueeze(0)  # [1, 128]

                            attention_scores_3d = self.smile_attention(drug_emb_3d)  # [num_valid_drugs, 1]
                            drug_emb_3d = (attention_scores_3d * drug_emb_3d).sum(dim=0).unsqueeze(0)  # [1, 128]




                        # fusion
                        drug_emb_fused = self.fusion(torch.cat([drug_emb_2d, drug_emb_3d], dim=-1))  # (1, 128)

                        # 2d 3d 复原函数
                        recon_drug_emb_2d = self.decoder_2d(drug_emb_fused)
                        recon_drug_emb_3d = self.decoder_3d(drug_emb_fused)
                        
                        
                        drug_emb_2d_origin_list.append(drug_emb_2d)
                        drug_emb_3d_origin_list.append(drug_emb_3d)
                        drug_emb_2d_recon_list.append(recon_drug_emb_2d)
                        drug_emb_3d_recon_list.append(recon_drug_emb_3d)
                        drug_emb_list[i].append(drug_emb_fused)
                # list of lists to list of tensors
                
                
                stacked_list = [
                    torch.cat(sublist, dim=0) if sublist else torch.empty(0, 128)  
                    for sublist in drug_emb_list
                ]
                x = torch.nn.utils.rnn.pad_sequence(stacked_list, batch_first=True, padding_value=0.0)
                x = x.to(device=self.device)
                # print(f"填充后的张量形状：{x.shape}")
                # x：(patient, visit, embedding_dim)
            output, hidden = self.visit_gru[feature_key](x) # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)
            patient_emb_list.append(hidden.squeeze(dim=0))  #（2,128）

        # """处理drug"""
        # drugs_hist = batch_data['drugs_hist']
        # drug_emb_list = []
        # drug_emb_2d_origin_list = []
        # drug_emb_3d_origin_list = []
        # drug_emb_2d_recon_list = []
        # drug_emb_3d_recon_list = []

        # for i in range(batch_size):
        #     drug_hist = drugs_hist[i][0]  # 获取每个病人的药物历史
        #     if len(drug_hist) == 0:
        #         drug_emb_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
        #         drug_emb_2d_origin_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
        #         drug_emb_3d_origin_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
        #         drug_emb_2d_recon_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
        #         drug_emb_3d_recon_list.append(torch.zeros(1, self.embedding_dim, device=self.device))
        #         continue
        #     drug_smiles = get_smiles_list(drug_hist)  # 获取药物的SMILES列表
        #     molecule_graph = StaticParaDict(
        #         **graph_batch_from_smiles(drug_smiles, self.device)
        #     )


        #     # todo 这里为什么是重新计算2d 3d的表征，而不是从上面的embeddings_2D和embeddings_3D中直接抽取
        #     # 2d
        #     drug_emb_2d = self.molecule_encoder(molecule_graph)
        #     attention_scores_2d = self.smile_attention(drug_emb_2d)  # (num_drugs, 1)
        #     drug_emb_2d = (attention_scores_2d * drug_emb_2d).sum(dim=0).unsqueeze(0) # 1,128

        #     # 3d
        #     druggeolist = [self.drug_sdf_dict[self.smiles2cid[x]].to(self.device) for x in drug_smiles]
        #     druggeoloader = loader.DataLoader(dataset=druggeolist, batch_size=len(druggeolist), shuffle=False)
        #     drug_emb_3d = self.drug_model(next(iter(druggeoloader)))  # (num_drugs, drug_emb_dim)
        #     drug_emb_3d = self.drug_fc(drug_emb_3d)     # (num_drugs, 128)
        #     attention_scores_3d = self.smile_attention(drug_emb_3d)  # (num_drugs, 1)
        #     drug_emb_3d = (attention_scores_3d * drug_emb_3d).sum(dim=0).unsqueeze(0)

        #     # fusion
        #     drug_emb_fused = self.fusion(torch.cat([drug_emb_2d, drug_emb_3d], dim=-1))  # (1, 128)

        #     # 2d 3d 复原函数
        #     recon_drug_emb_2d = self.decoder_2d(drug_emb_fused)
        #     recon_drug_emb_3d = self.decoder_3d(drug_emb_fused)
            
            
        #     drug_emb_2d_origin_list.append(drug_emb_2d)
        #     drug_emb_3d_origin_list.append(drug_emb_3d)
        #     drug_emb_2d_recon_list.append(recon_drug_emb_2d)
        #     drug_emb_3d_recon_list.append(recon_drug_emb_3d)
        #     drug_emb_list.append(drug_emb_fused)


        drug_emb_2d_origin_list = torch.cat(drug_emb_2d_origin_list, dim=0)
        drug_emb_3d_origin_list = torch.cat(drug_emb_3d_origin_list, dim=0)
        drug_emb_2d_recon_list = torch.cat(drug_emb_2d_recon_list, dim=0)
        drug_emb_3d_recon_list = torch.cat(drug_emb_3d_recon_list, dim=0)

        L_2D = contrastive_loss(drug_emb_2d_recon_list, drug_emb_2d_origin_list, temperature=0.07)
        L_3D = contrastive_loss(drug_emb_3d_recon_list, drug_emb_3d_origin_list, temperature=0.07)
        L_cvc = 0.5 * (L_2D + L_3D)

        # patient_emb_list.append(drug_emb_list)  # (2,128)

        patient_emb = torch.cat(patient_emb_list, dim=-1) #(2,128*3)

        repr_query_last = self.drug_query(patient_emb)   # 2*195
        logits_list = []
        for i in range(batch_size):
            repr_weight_last_i = repr_query_last[i].unsqueeze(0)     # (1, 195)  
            diag_repr_weight_last_i = torch.diag(torch.sigmoid(repr_weight_last_i.squeeze(0)))  # (195, 195)
            Atten_i = torch.matmul(diag_repr_weight_last_i, molecule_embeddings)  # (195, 128)注意力（patient_emb，molecule_embedding）
            logits_i = self.score_extractor(Atten_i).t()  # (195, 1) -> (1, 195)
            logits_list.append(logits_i)
        logits = torch.cat(logits_list, dim=0)  # (2, 195)

        return logits, L_cvc
