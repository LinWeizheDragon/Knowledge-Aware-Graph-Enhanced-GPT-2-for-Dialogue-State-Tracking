"""
graph_model.py: GAT model for the paper:
Lin, W., Tseng, B. H., & Byrne, B. (2021). Knowledge-Aware Graph-Enhanced GPT-2 for Dialogue State Tracking. EMNLP 2021.
https://arxiv.org/abs/2104.04466v3
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_modules import GraphFilterBatchAttentional

class GraphModel(nn.Module):
    """
        Graph Models used in paper
    """
    def __init__(self, config):
        super(GraphModel,self).__init__()
        self.config = config
        self.graph_layers = nn.ModuleList()
        # Read hyperparameters from the global config
        num_layer = self.config.model_config.graph_model.num_layer
        G = self.config.model_config.graph_model.feature_size
        F = self.config.model_config.graph_model.feature_size
        P = self.config.model_config.graph_model.num_head
        K = self.config.model_config.graph_model.num_hop

        for _ in range(num_layer):
            self.graph_layers.append(
                GraphFilterBatchAttentional(G=G, 
                                            F=F, 
                                            K=K, 
                                            P=P, 
                                            concatenate=False, 
                                            bias=False)
            )
        self.dropout = nn.Dropout(self.config.model_config.graph_model.dropout)

    def add_GSO(self, S):
        """Add GSO (ontology descriptor)

        Args:
            S (Tensor): ontology descriptor B x E x N x N
        """
        for graph_layer in self.graph_layers:
            graph_layer.addGSO(S)

    def get_GSO(self):
        attentions = []
        for graph_layer in self.graph_layers:
            attentions += [graph_layer.returnAttentionGSO()]
        return attentions

    def forward(self,x):
        y = x
        for graph_layer in self.graph_layers:
            y = graph_layer(y)
        y = self.dropout(y)
        return y
