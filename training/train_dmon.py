from models.gcn_dmon import GCNDMoN
import utils.graph_utils as graph_utils
from utils import metrics
import torch.optim as optim
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def train(adjacency, features, prior_membership=None, model = None, architecture = [64], 
              collapse_regularization = 0.2, sparsity_regularization = 0.0, 
              knowledge_regularization = 0.0, connectivity_regularization = 0.0,
              dropout_rate = 0.2, n_clusters = 40, initialization = "louvain", clustering_method = "auto", activation = "softmax",
              n_epochs = 3000, learning_rate = 0.001, plot=False, patience = 3000, verbose = False, device="cpu"):
    
    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]

    # Convert to PyTorch sparse tensors
    graph_ = graph_utils.convert_scipy_sparse_to_sparse_tensor(adjacency).to(device)
    graph_normalized = graph_utils.convert_scipy_sparse_to_sparse_tensor(
        graph_utils.normalize_graph(adjacency.copy())).to(device)

    # Print shapes using correct attributes for SparseTensor
    # print(f"Labels shape: {labels.shape}, label indices shape: {label_indices.shape}")
    # print(f"Graph sparse sizes: {graph_.sparse_sizes()}, graph normalized sparse sizes: {graph_normalized.sparse_sizes()}")
        
    # Create model
    if model is None:
        model = GCNDMoN(
            architecture=[feature_size] + architecture,
            n_clusters=n_clusters,
            collapse_regularization=collapse_regularization,
            knowledge_regularization=knowledge_regularization,
            connectivity_regularization=connectivity_regularization,
            sparsity_regularization=sparsity_regularization,
            initialization=initialization,
            activation=activation,
            dropout_rate=dropout_rate
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if verbose: print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
        # initialize with louvain
        if initialization == "louvain":
            print("Initializing with Louvain community detection...")
            model.initialize_with_louvain(features, graph_normalized, graph_)
            louvain_assignments = model.initialize_with_louvain(
                features=features,
                adjacency=graph_,
                norm_adjacency=graph_normalized,
                device=device,
                lr=learning_rate,
                epochs=3000,
                patience=20,
                verbose=verbose
            )
    else:
        # reset model parameters
        print("reset model parameters...")
        model.dmon.collapse_regularization = collapse_regularization
        model.dmon.knowledge_regularization = knowledge_regularization
        model.dmon.connectivity_regularization = connectivity_regularization
        model.dmon.sparsity_regularization = sparsity_regularization

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize loss tracking lists
    spectral_losses = []
    collapse_losses = []
    knowledge_losses = []
    connectivity_losses = []
    sparsity_losses = []
    total_losses = []

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    
    # Training loop
    # print("Training for", n_epochs, "epochs...")
    # if prior_membership is not None:
    # print("prior_membership", prior_membership[0])
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        _, assignments, embeddings = model(features, graph_normalized, graph_, prior_membership=prior_membership)
        
        # Get losses
        losses = model.get_losses()
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value 

        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        spectral_losses.append(losses['spectral_loss'].item())
        collapse_losses.append(losses['collapse_loss'].item())
        knowledge_losses.append(losses['knowledge_loss'].item())
        connectivity_losses.append(losses['connectivity_loss'].item())
        sparsity_losses.append(losses['sparsity_loss'].item())
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
                print(f'Epoch {epoch}, Spectral Loss: {losses['spectral_loss'].item():.4f}, '
                    f'Collapse Loss: {losses['collapse_loss'].item():.4f}, '
                    f'Knowledge Loss: {losses['knowledge_loss'].item():.4f}, '
                    f'Sparsity Loss: {losses['sparsity_loss'].item():.4f}, '
                    f'Total Loss: {total_loss.item():.4f}')

    # Load the best model state before evaluation
    model.load_state_dict(best_model_state)

    # Obtain the cluster assignments
    model.eval()
    with torch.no_grad():
        _, assignments, embeddings = model(features, graph_normalized, graph_, prior_membership=prior_membership)
        clusters = assignments.cpu().numpy().argmax(axis=1)  # Convert soft to hard clusters

    # Print metrics
    modularity = metrics.modularity(adjacency, clusters)
    if verbose: print(f'Modularity: {modularity:.4f}, Conductance: {metrics.conductance(adjacency, clusters):.4f}')

    print("clusters", clusters.shape)
    print(clusters)
    
    # Plot losses
    if plot and n_epochs > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(spectral_losses, label='Spectral Loss')
        if collapse_regularization > 0:
            plt.plot(collapse_losses, label='Collapse Loss')
        if knowledge_regularization > 0:
            plt.plot(knowledge_losses, label='Knowledge Loss')
        if connectivity_regularization > 0:
            plt.plot(connectivity_losses, label='Connectivity Loss')
        plt.plot(total_losses, label='Total Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curves (modularity={modularity:.4f})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return clusters, model, modularity, assignments