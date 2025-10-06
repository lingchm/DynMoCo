import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from layers.gcn import GCNConv
from layers.dmon import DMoNPooling
from utils.louvain import louvain_community_detection, compute_modularity


def initialize_from_louvain(model, adjacency, features, norm_adjacency, device='cpu', seed=123, 
                           lr=0.01, epochs=100, patience=20, verbose=True):
    """
    Train the entire GCNDMoN model to match Louvain community detection assignments.
    
    Args:
        model: GCNDMoN model instance
        adjacency: Adjacency matrix for Louvain initialization
        features: Node features for Louvain initialization
        norm_adjacency: Normalized adjacency matrix for GCN layers
        device: Device to place tensors on
        seed: Random seed for reproducible Louvain results
        lr: Learning rate for training
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        verbose: Whether to print training progress
    
    Returns:
        trained_model: The trained GCNDMoN model
        louvain_assignments: The target Louvain assignment matrix
    """
    # Get Louvain partition
    partition, n_communities = louvain_community_detection(adjacency, model.n_clusters, seed=seed)
    
    # Convert to tensor assignment matrix
    n_nodes = features.shape[0]
    louvain_assignments = torch.zeros(n_nodes, model.n_clusters, device=device)
    
    for node, community in partition.items():
        if community < model.n_clusters:  # Ensure we don't exceed target clusters
            louvain_assignments[node, community] = 1.0
    
    # Normalize assignments to sum to 1 for each node
    row_sums = louvain_assignments.sum(dim=1, keepdim=True)
    louvain_assignments = louvain_assignments / (row_sums + 1e-8)
    
    # Move model to device and set to training mode
    model = model.to(device)
    model.train()
    
    # Setup optimizer for all model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss function for assignment matching (KL divergence)
    def assignment_loss(pred_assignments, target_assignments):
        # Add small epsilon to avoid log(0)
        pred_assignments = torch.clamp(pred_assignments, min=1e-8, max=1.0)
        target_assignments = torch.clamp(target_assignments, min=1e-8, max=1.0)
        
        # KL divergence: KL(target || pred)
        kl_loss = torch.sum(target_assignments * torch.log(target_assignments / pred_assignments))
        
        # Also add MSE loss for better convergence
        mse_loss = F.mse_loss(pred_assignments, target_assignments)
        
        return mse_loss + kl_loss
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    if verbose:
        print(f"Training GCNDMoN model to match Louvain assignments for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass through the model
        pool, pool_assignment = model(features, norm_adjacency, adjacency)
        
        # Compute assignment matching loss
        assignment_matching_loss = assignment_loss(pool_assignment, louvain_assignments)
        
        # Get DMoN losses
        dmon_losses = model.get_losses()
        total_dmon_loss = sum(dmon_losses.values())
        
        # Total loss: assignment matching + DMoN regularization
        total_loss = assignment_matching_loss + 0.1 * total_dmon_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
            
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.6f}, "
                  f"Assignment Loss: {assignment_matching_loss.item():.6f}, "
                  f"DMoN Loss: {total_dmon_loss.item():.6f}")
    
    if verbose:
        print(f"Training completed. Best loss: {best_loss:.6f}")
    
    # Set model to evaluation mode
    model.eval()
    
    return model, louvain_assignments


class GCNDMoN(torch.nn.Module):
    """Simple DMoN model without torch_sparse dependency."""

    def __init__(self, architecture, n_clusters, 
                 collapse_regularization, knowledge_regularization=0.0, 
                 sparsity_regularization=0.0, connectivity_regularization=0.0,
                 dropout_rate=0.2, initialization="louvain", activation="softmax"):
        super(GCNDMoN, self).__init__()
        self.architecture = architecture
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.knowledge_regularization = knowledge_regularization
        self.sparsity_regularization = sparsity_regularization
        self.connectivity_regularization = connectivity_regularization
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Initialize GCN layers based on architecture
        self.gcn_layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            gcn_layer = GCNConv(in_dim, out_dim)  # Your corrected GCN class
            self.gcn_layers.append(gcn_layer)
            self.bns.append(torch.nn.BatchNorm1d(out_dim))
            
        self.dmon = DMoNPooling(
            n_clusters=n_clusters,
            collapse_regularization=collapse_regularization,
            knowledge_regularization=knowledge_regularization,
            sparsity_regularization=sparsity_regularization,
            connectivity_regularization=connectivity_regularization,
            activation=activation,
            dropout_rate=dropout_rate
        )
        self.dmon.reset_parameters(architecture[-1])
        self.dropout = torch.nn.Dropout(dropout_rate)

    def initialize_with_louvain(self, features, norm_adjacency, adjacency, device='cpu', seed=123, 
                               lr=0.001, epochs=3000, patience=20, verbose=True):
        """Initialize and train the model using Louvain community detection.
        
        Args:
            adjacency: Adjacency matrix for Louvain initialization
            features: Node features for Louvain initialization
            norm_adjacency: Normalized adjacency matrix for GCN layers
            device: Device to place tensors on
            seed: Random seed for reproducible Louvain results
            lr: Learning rate for training
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            verbose: Whether to print training progress
        """
        trained_model, louvain_assignments = initialize_from_louvain(
            self, adjacency, features, norm_adjacency, device, seed, lr, epochs, patience, verbose
        )
        
        # Update the current model with trained parameters
        self.load_state_dict(trained_model.state_dict())
        
        return louvain_assignments

    def forward(self, features, norm_adjacency, adjacency, comembership=None, prior_membership=None):
        
        output = features
        
        for l in range(len(self.gcn_layers)):
            output = self.gcn_layers[l](output, norm_adjacency)
            output = self.bns[l](output)
            output = F.selu(output)
            output = self.dropout(output)
            
        pool, pool_assignment = self.dmon(output, adjacency, comembership, prior_membership)
        
        return pool, pool_assignment, output
        
    def get_losses(self):
        return self.dmon.get_losses()