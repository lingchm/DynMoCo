"""PyTorch implementation of DiffPooling training script from dmon/train_diffpool.py."""
import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models.gcn_diffpool import GCNDiffPool
import utils.graph_utils as graph_utils
import utils.metrics as metrics

def train(adjacency, features, model = None, architecture = [64], link_regularization = 1, ent_regularization = 1,
              dropout_rate = 0.2, n_clusters = 40, n_epochs = 3000, learning_rate = 0.001, plot=False, patience = 3000, verbose = False, device="cpu"):
    
    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]

    # Convert to PyTorch sparse tensors
    graph_ = graph_utils.convert_scipy_sparse_to_sparse_tensor(adjacency).to(device)
    graph_normalized = graph_utils.convert_scipy_sparse_to_sparse_tensor(
        graph_utils.normalize_graph(adjacency.copy())).to(device)

    # Create model
    if model is None:
        model = GCNDiffPool(
            architecture=[feature_size] + architecture,
            n_clusters=n_clusters,
            dropout_rate=dropout_rate,
            link_regularization=link_regularization,
            ent_regularization=ent_regularization
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if verbose: print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize loss tracking lists
    link_losses = []
    ent_losses = []
    total_losses = []

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        embeddings, assignments = model(features, graph_normalized, graph_)
        
        # Get losses
        losses = model.get_losses()
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value 

        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        link_losses.append(losses['link_loss'].item())
        ent_losses.append(losses['ent_loss'].item())
        total_losses.append(total_loss.item())
        
        # Early stopping logic
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Check for early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                print(f"Best loss: {best_loss:.6f}")
            break
        
        if verbose:
        # if epoch == n_epochs - 1:
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f'Epoch {epoch}, Link Loss: {losses['link_loss'].item():.4f}, '
                    f'Entropy Loss: {losses['ent_loss'].item():.4f}, '
                    f'Total Loss: {total_loss.item():.4f}')

    # Load the best model state before evaluation
    model.load_state_dict(best_model_state)

    # Obtain the cluster assignments
    model.eval()
    with torch.no_grad():
        embeddings, assignments = model(features, graph_normalized, graph_)
        clusters = assignments.cpu().numpy().argmax(axis=1)  # Convert soft to hard clusters

    # Print metrics
    modularity = metrics.modularity(adjacency, clusters)
    if verbose: print(f'Modularity: {modularity:.4f}, Conductance: {metrics.conductance(adjacency, clusters):.4f}')

    print("clusters", clusters.shape)
    print(clusters)
    
    # Plot losses
    if plot and n_epochs > 0:
        plt.figure(figsize=(10, 6))
        if link_regularization > 0:
            plt.plot(link_losses, label='Link Loss')
        if ent_regularization > 0:
            plt.plot(ent_losses, label='Entropy Loss')
        plt.plot(total_losses, label='Total Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curves (modularity={modularity:.4f})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return clusters, model, modularity, assignments