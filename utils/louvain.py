import numpy as np
import networkx as nx

def compute_modularity(adjacency, partition):
    """
    Compute modularity of a partition.
    
    Args:
        adjacency: Adjacency matrix (dense tensor or numpy array)
        partition: Dictionary mapping node indices to community assignments
    
    Returns:
        modularity: Modularity score
    """
    if hasattr(adjacency, 'to_dense'):
        adj_dense = adjacency.to_dense().cpu().numpy()
    else:
        adj_dense = adjacency.cpu().numpy() if hasattr(adjacency, 'cpu') else adjacency
    
    n_nodes = adj_dense.shape[0]
    total_edges = np.sum(adj_dense) / 2  # Undirected graph, so divide by 2
    
    if total_edges == 0:
        return 0.0
    
    modularity = 0.0
    
    # Compute degrees
    degrees = np.sum(adj_dense, axis=1)
    
    # For each pair of nodes
    for i in range(n_nodes):
        for j in range(n_nodes):
            if partition[i] == partition[j]:  # Same community
                # Expected number of edges between i and j
                expected_edges = degrees[i] * degrees[j] / (2 * total_edges)
                # Actual number of edges between i and j
                actual_edges = adj_dense[i, j]
                # Contribution to modularity
                modularity += actual_edges - expected_edges
    
    modularity /= (2 * total_edges)
    return modularity


def louvain_community_detection(adjacency, n_clusters=None, seed=123, verbose=True):
    """
    Perform Louvain community detection on the given adjacency matrix.
    
    Args:
        adjacency: SparseTensor or dense tensor representing the adjacency matrix
        n_clusters: Target number of clusters (optional, if None uses Louvain's natural clustering)
        seed: Random seed for reproducible results
    
    Returns:
        partition: Dictionary mapping node indices to community assignments
        n_communities: Number of communities found
    """
    # Convert adjacency to networkx graph
    if hasattr(adjacency, 'to_dense'):
        adj_dense = adjacency.to_dense().cpu().numpy()
    else:
        adj_dense = adjacency.cpu().numpy()
    
    # Create networkx graph
    G = nx.from_numpy_array(adj_dense)
    
    # Run Louvain algorithm using NetworkX's built-in function
    communities = nx.community.louvain_communities(G, seed=seed)
    
    # Convert partition to node-to-community mapping
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i
    
    n_communities = len(communities)
    
    # Compute modularity of the original Louvain partition
    original_modularity = compute_modularity(adjacency, node_to_community)
    if verbose: print(f"Original Louvain partition modularity: {original_modularity:.4f} num_clusters: {n_communities}")
    
    # If target number of clusters is specified and different from found communities
    if n_clusters is not None and n_communities > n_clusters:
        # Split largest communities
        community_sizes = {i: len(comm) for i, comm in enumerate(communities)}
        while n_communities < n_clusters:
            # Find largest community to split
            largest_comm = max(community_sizes.items(), key=lambda x: x[1])[0]
            comm_nodes = list(communities[largest_comm])
            
            if len(comm_nodes) < 2:
                break  # Can't split further
            
            # Simple split: take first half of nodes
            split_point = len(comm_nodes) // 2
            new_comm_nodes = comm_nodes[split_point:]
            
            # Create new community
            new_comm_id = n_communities
            for node in new_comm_nodes:
                node_to_community[node] = new_comm_id
            
            # Update communities
            communities[largest_comm] = set(comm_nodes[:split_point])
            communities[new_comm_id] = set(new_comm_nodes)
            
            n_communities += 1
        
        # Compute modularity after adjustment
        adjusted_modularity = compute_modularity(adjacency, node_to_community)
        print(f"Adjusted partition modularity: {adjusted_modularity:.4f} Adjusted num_clusters: {n_communities}")
    
    return node_to_community, n_communities

