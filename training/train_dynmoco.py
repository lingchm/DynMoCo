
from models.dynmoco import DynMoCo
import utils.metrics as metrics
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt


def train(dataloader, residue_indices, comembership=None, architecture = [64], 
              collapse_regularization = 0.2, knowledge_regularization = 0.1, sparsity_regularization = 0.0, 
              dropout_rate = 0.0, n_clusters = 40, initialization = "louvain", activation = "softmax", gumbel_tau=0.5,
              n_epochs = 3000, learning_rate = 0.001, plot=False, patience = 3000, verbose = True, device="cpu"):
    
    
    # ensure reproducibility 
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

    # Create model
    model = DynMoCo(
        architecture=architecture, n_clusters=n_clusters,
        collapse_regularization=collapse_regularization, 
        knowledge_regularization=knowledge_regularization, 
        sparsity_regularization=sparsity_regularization, 
        dropout_rate=dropout_rate, activation=activation, 
        initialization=initialization, gumbel_tau=gumbel_tau, device=device
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose: print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

    # Initialize loss tracking lists
    spectral_losses = []
    collapse_losses = []
    knowledge_losses = []
    sparsity_losses = []
    total_losses = []

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    # print("Training for", n_epochs, "epochs...")
    for epoch in range(n_epochs):
        
        model.train()
        optimizer.zero_grad()
        
        for batch in dataloader:
            
            graph_list = batch['graph_list'][0]  # Raw adjacency graphs (sparse tensors)
            norm_graph_list = batch['norm_graphs_list'][0]  # Normalized adjacency graphs (sparse tensors)
            features_list = batch['features_list'][0]  # Node features (dense tensors)
            comembership = batch['comembership'][0]
            
            # Forward pass
            assignments_list, embeddings_list = model(features_list, norm_graph_list, graph_list, comembership)

            # Get losses
            losses = model.get_losses()
            total_loss = 0
            for loss_name, loss_value in losses.items():
                total_loss += loss_value 

            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
        
        # Store losses
        if epoch > 5:
            spectral_losses.append(losses['spectral_loss'].item())
            collapse_losses.append(losses['collapse_loss'].item())
            knowledge_losses.append(losses['knowledge_loss'].item())
            sparsity_losses.append(losses['sparsity_loss'].item())
            total_losses.append(total_loss.item())
        
        # Early stopping logic
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                print(f"Best loss: {best_loss:.6f}")
            break
        
        if verbose:
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f'Epoch {epoch}, Spectral Loss: {losses['spectral_loss'].item():.4f}, '
                    # f'Collapse Loss: {losses['collapse_loss'].item():.4f}, '
                    f'Knowledge Loss: {losses['knowledge_loss'].item():.4f}, '
                    f'Sparsity Loss: {losses['sparsity_loss'].item():.4f}, '
                    f'Total Loss: {total_loss.item():.4f}')

    # Plot losses
    if plot and n_epochs > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(spectral_losses, label='Spectral Loss')
        if collapse_regularization > 0:
            plt.plot(collapse_losses, label='Collapse Loss')
        if knowledge_regularization > 0:
            plt.plot(knowledge_losses, label='Knowledge Loss')
        if sparsity_regularization > 0:
            plt.plot(sparsity_losses, label='Sparsity Loss')
        plt.plot(total_losses, label='Total Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Load the best model state before evaluation
    model.load_state_dict(best_model_state)

    # Obtain the cluster assignments
    communities_batch = []
    modularity_batch = defaultdict(list)
    conductance_batch = defaultdict(list)
    model.eval()
    with torch.no_grad():
        
        for batch in dataloader:
            graph_list = batch['graph_list'][0]  
            norm_graph_list = batch['norm_graphs_list'][0]  
            features_list = batch['features_list'][0]  
            adjacency_list = batch['adjacency_list'][0]
            comembership = batch['comembership'][0]
            
            assignments_list, embeddings_list = model(features_list, norm_graph_list, graph_list, comembership)
            
            min_nodes = 5
            communities = defaultdict(dict)
            for t in range(len(graph_list)):
                assignments = assignments_list[t].cpu().numpy()
                clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters
                # Print metrics
                modularity = metrics.modularity(adjacency_list[t], clusters)
                conductance = metrics.conductance(adjacency_list[t], clusters)
                modularity_batch[t].append(modularity)
                conductance_batch[t].append(conductance)
                
                # exclude communities with too few nodes 
                unique, counts = np.unique(clusters, return_counts=True)
                communities_included = list(unique[counts >= min_nodes])
                mask = np.isin(clusters, communities_included)
                filtered_clusters = clusters[mask]
                filtered_nodes = np.nonzero(mask)[0]  # indices of kept nodes
                communities_included = [str(c) for c in communities_included]

                # if verbose: print(f't={t}: {len(communities_included)} communities, Modularity: {np.mean(modularity_batch[t]):.4f}, Conductance: {np.mean(conductance_batch[t]):.4f}')

                # store data 
                for i in range(len(clusters)):
                    node = residue_indices[i]
                    cluster = str(clusters[i])
                    if cluster in communities_included:
                        if node in communities[str(cluster)]:
                            communities[str(cluster)][node].append(t)
                        else:
                            communities[str(cluster)][node] = [t]
            communities_batch.append(communities)
    
    for t in range(len(graph_list)):
        print(f't={t}: Modularity={np.mean(modularity_batch[t]):.3f}+-{np.std(modularity_batch[t]):.3f}, Conductance={np.mean(conductance_batch[t]):.4f}+-{np.std(conductance_batch[t]):.3f}')

    return communities_batch, model