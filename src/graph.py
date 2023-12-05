import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os

from torch_geometric.nn import GCNConv, GATConv

class GraphLayer(nn.Module):
    def __init__(self, config, graph_type):
        super(GraphLayer, self).__init__()
        self.config = config

        self.graph_type = graph_type
        if self.graph_type == 'GCN':
            self.graph = GCNConv(config.hidden_size, config.hidden_size)
        elif self.graph_type == 'GAT':
            self.graph = GATConv(config.hidden_size, config.hidden_size, 1)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.hidden_dropout_prob
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, label_emb, extra_attn):
        residual = label_emb
        if self.graph_type == 'GCN' or self.graph_type == 'GAT':
            label_emb = self.graph(label_emb.squeeze(0), edge_index=extra_attn)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)
        else:
            raise NotImplementedError
        return label_emb

class GraphEncoder(nn.Module):
    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEncoder, self).__init__()
        self.config = config
        
        self.hir_layers = nn.ModuleList([GraphLayer(config, graph_type) for _ in range(layer)])

        self.label_num = config.num_labels - 3
        self.graph_type = graph_type

        self.path_list = nn.Parameter(torch.tensor(path_list).transpose(0, 1), requires_grad=False)

    def forward(self, label_emb, embeddings):
        extra_attn = None
        if self.graph_type == 'GCN' or self.graph_type == 'GAT':
            extra_attn = self.path_list

        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb.unsqueeze(0), extra_attn)

        # return label_emb.squeeze(0)
        return label_emb.squeeze()
    
class MyGraphLayer(nn.Module):
    def __init__(self, graph_type, embedding_size):
        super(MyGraphLayer, self).__init__()

        if graph_type == 'GCN':
            self.conv = GCNConv(embedding_size, embedding_size)
        elif graph_type == 'GAT':
            self.conv = GATConv(embedding_size, embedding_size)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

class MyGraphEncoder(nn.Module):
    def __init__(self, graph_type, layers, path_list, embedding_size):
        super(MyGraphEncoder, self).__init__()
        self.graph_layers = nn.ModuleList([MyGraphLayer(graph_type, embedding_size) for _ in range(layers)])
        self.edge_index = nn.Parameter(torch.tensor(path_list).transpose(0, 1), requires_grad=False)
        
    def forward(self, x):
        for layer in self.graph_layers:
            x = layer(x, self.edge_index)
        return x
