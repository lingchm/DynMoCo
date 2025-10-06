"""PyTorch implementation of DMoN training.

This is the PyTorch implementation of the DMoN training script from the original
TensorFlow code.
"""
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import normalized_mutual_info_score
from models.gcn_dmon import GCNDMoN
import utils
import utils.metrics as metrics


def main():
    parser = argparse.ArgumentParser(description='Train DMoN with PyTorch')
    parser.add_argument('--graph_path', type=str, required=True,
                       help='Input graph path')
    parser.add_argument('--architecture', nargs='+', type=int, default=[64],
                       help='Network architecture')
    parser.add_argument('--collapse_regularization', type=float, default=1.0,
                       help='Collapse regularization')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                       help='Dropout rate for GNN representations')
    parser.add_argument('--n_clusters', type=int, default=16,
                       help='Number of clusters')
    parser.add_argument('--n_epochs', type=int, default=1000,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and process the data
    adjacency, features, labels, label_indices = utils.load_npz(args.graph_path)
    features = torch.FloatTensor(features.todense()).to(device)
    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]
    
    # Convert to PyTorch sparse tensors
    graph = utils.convert_scipy_sparse_to_sparse_tensor(adjacency).to(device)
    graph_normalized = utils.convert_scipy_sparse_to_sparse_tensor(
        utils.normalize_graph(adjacency.copy())).to(device)
    
    # Print shapes using correct attributes for SparseTensor
    print(f"Data shape: {features.shape}, adjacency shape: {adjacency.shape}")
    print(f"Labels shape: {labels.shape}, label indices shape: {label_indices.shape}")
    print(f"Graph sparse sizes: {graph.sparse_sizes()}, graph normalized sparse sizes: {graph_normalized.sparse_sizes()}")
    
    # Create model
    model = GCNDMoN(
        architecture=[feature_size] + args.architecture,
        n_clusters=args.n_clusters,
        collapse_regularization=args.collapse_regularization,
        dropout_rate=args.dropout_rate
    ).to(device)
    print(model)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        _, assignments = model(features, graph_normalized, graph)
        
        # Get losses
        spectral_loss, collapse_loss = model.get_losses()
        total_loss = spectral_loss + collapse_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == args.n_epochs - 1:
            print(f'Epoch {epoch}, Spectral Loss: {spectral_loss.item():.4f}, '
                  f'Collapse Loss: {collapse_loss.item():.4f}, '
                  f'Total Loss: {total_loss.item():.4f}')
    
    # Obtain the cluster assignments
    model.eval()
    with torch.no_grad():
        _, assignments = model(features, graph_normalized, graph)
        assignments = assignments.cpu().numpy()
        clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters
    
    # Print metrics
    print('Conductance:', metrics.conductance(adjacency, clusters))
    print('Modularity:', metrics.modularity(adjacency, clusters))
    print('NMI:', normalized_mutual_info_score(
        labels, clusters[label_indices], average_method='arithmetic'))
    
    precision = metrics.pairwise_precision(labels, clusters[label_indices])
    recall = metrics.pairwise_recall(labels, clusters[label_indices])
    print('F1:', 2 * precision * recall / (precision + recall))


if __name__ == '__main__':
    main() 