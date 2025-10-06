import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.gcn import GCNConv
from layers.dmon import DMoNPooling
from layers.egcnh import EGCUH
from layers.hgru import HGRUCell
from layers.summarize import SummarizeLayer
from collections import defaultdict
from utils.louvain import louvain_community_detection

class DynMoCo(torch.nn.Module):
    """Simple DMoN model without torch_sparse dependency."""
    
    def __init__(self, architecture, n_clusters, 
                 collapse_regularization=0.0, knowledge_regularization=0.0, 
                 connectivity_regularization=0.0, sparsity_regularization=0.0, 
                 dropout_rate=0.2, initialization="louvain", activation="softmax", 
                 gumbel_tau=0.5, device="cpu"):
        
        super(DynMoCo, self).__init__()
        self.n_clusters = n_clusters
        self.dropout_rate = dropout_rate
        self.losses = defaultdict(int)
        
        # Initialize GCN layers based on architecture
        self.gcn_layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            gcn_layer = GCNConv(in_dim, out_dim)  # Your corrected GCN class
            self.gcn_layers.append(gcn_layer)
            self.bns.append(torch.nn.BatchNorm1d(out_dim))
        
        # Initialize pooling layer
        self.gru_cell = nn.GRUCell(input_size=architecture[-1], hidden_size=n_clusters)
        
        # Initialize pooling layer
        self.dmon = DMoNPooling(
            n_clusters=n_clusters,
            collapse_regularization=collapse_regularization,
            knowledge_regularization=knowledge_regularization,
            connectivity_regularization=connectivity_regularization,
            sparsity_regularization=sparsity_regularization,
            activation=activation, gumbel_tau=gumbel_tau,
            dropout_rate=dropout_rate
        )
        self.dmon.reset_parameters(architecture[-1])
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.initialization = initialization
        self.device = device
        self.initialization_weights = None 
        self.activation = activation
        self.gumbel_tau = gumbel_tau

    def initialize_from_louvain(self, adjacency, features, norm_adjacency, seed=1234, verbose=False):
        # Get Louvain partition
        partition, n_communities = louvain_community_detection(adjacency, self.n_clusters, seed=seed, verbose=False)
        
        # Convert to tensor assignment matrix
        n_nodes = features.shape[0]
        louvain_assignments = torch.zeros(n_nodes, self.n_clusters, device=self.device)
        
        for node, community in partition.items():
            if community < self.n_clusters:  # Ensure we don't exceed target clusters
                louvain_assignments[node, community] = 1.0
        
        # Normalize assignments to sum to 1 for each node
        row_sums = louvain_assignments.sum(dim=1, keepdim=True)
        louvain_assignments = louvain_assignments / (row_sums + 1e-8)
        self.initialization_weights = louvain_assignments
    
    def embed(self, features, norm_adjacency):
        output = features
        for l in range(len(self.gcn_layers)):
            output = self.gcn_layers[l](output, norm_adjacency)
            output = self.bns[l](output)
            output = F.selu(output)
            output = self.dropout(output)
        return output 
 
    def evolve(self, embeddings, assignments):
        logits = self.gru_cell(embeddings, assignments)
        return logits 

    def initialize_hidden_state(self, features, norm_adjacency, adjacency):
        if self.initialization_weights is None:
            if self.initialization == "louvain":
                # Use Louvain initialization
                self.initialize_from_louvain(adjacency, features, norm_adjacency, verbose=True)
            else:
                # Use orthogonal initialization
                self.initialization_weights = torch.zeros(features.shape[0], self.n_clusters)
                nn.init.orthogonal_(self.initialization_weights)

    def forward(self, features_list, norm_adjacency_list, adjacency_list, comembership=None):
        """
        Args:
            features_list: List of features for each layer
            norm_adjacency_list: List of normalized adjacency matrices for each layer
            adjacency_list: List of adjacency matrices for each layer
            comembership: Comembership matrix for the current layer
        """
        assignment_list, output_list = [], []
        total_losses = defaultdict(int)
        
        # initialize hidden state 
        self.initialize_hidden_state(features_list[0], norm_adjacency_list[0], adjacency_list[0])
        assignments = self.initialization_weights.clone()
        
        for i, (features, norm_adjacency, adjacency) in enumerate(zip(features_list, norm_adjacency_list, adjacency_list)):
        
            # feature extraction GCN
            embeddings = self.embed(features, norm_adjacency)
            
            # time-evolve pooling GCN 
            assignments_new = self.evolve(embeddings, assignments)

            # get community assignments 
            embeddings_pooled = self.dmon.pool(embeddings, adjacency, assignments_new, comembership, prior_membership=assignments)
            
            # accumulate losses over time 
            losses = self.dmon.get_losses()
            for k, v in losses.items():
                total_losses[k] = total_losses[k] + v
            
            assignment_list.append(assignments_new)
            output_list.append(embeddings)
            assignments = assignments_new
            
            self.losses = total_losses
        
        return assignment_list, output_list
        
    def get_losses(self):
        return self.losses