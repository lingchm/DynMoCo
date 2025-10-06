import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import utils.metrics as metrics

def find_optimal_clusters_per_timepoint(embs, max_clusters=10, plot=True):
    """
    Find optimal number of clusters for each time point separately.
    
    Parameters:
    -----------
    embs : list of numpy arrays
        List of embedding matrices, each of shape (N, d)
    max_clusters : int
        Maximum number of clusters to test
    plot : bool
        Whether to create visualization plots
        
    Returns:
    --------
    optimal_clusters_per_time : list
        List of optimal cluster counts for each time point
    all_silhouette_scores : list
        List of silhouette scores for each time point
    """
    
    optimal_clusters_per_time = []
    all_silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for t, emb in enumerate(embs):
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(emb)
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                score = silhouette_score(emb, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        all_silhouette_scores.append(silhouette_scores)
        optimal_clusters = np.argmax(silhouette_scores) + 2
        optimal_clusters_per_time.append(optimal_clusters)

    return optimal_clusters_per_time, all_silhouette_scores

# Function to get communities using time-specific optimal clusters
def get_communities_with_optimal_clusters(embs, optimal_clusters_per_time, index_to_residue=None, adjacency_list=None):
    """
    Get communities using the optimal number of clusters for each time point.
    
    Parameters:
    -----------
    embs : list of numpy arrays
        List of embedding matrices
    optimal_clusters_per_time : list
        List of optimal cluster counts for each time point
        
    Returns:
    --------
    communities : dict
        Dictionary with community labels as keys and node-time mappings as values
    """
    from collections import defaultdict
    
    communities = defaultdict(lambda: defaultdict(list))
    
    for t, (emb, n_clusters) in enumerate(zip(embs, optimal_clusters_per_time)):
        # Perform k-means clustering with optimal number of clusters for this time point
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb)
        if adjacency_list is not None:
            modularity = metrics.modularity(adjacency_list[t], labels)
            conductance = metrics.conductance(adjacency_list[t], labels)
            print(f't{t}: Modularity={modularity:.4f}, Conductance={conductance:.4f}')
            
        # For each node, record which community it belongs to at time t
        for node_id, community_label in enumerate(labels):
            community_key = f'C{community_label+1}'
            if index_to_residue is not None:
                communities[community_key][index_to_residue[node_id]].append(t)
            else:
                communities[community_key][node_id].append(t)
    
    return dict(communities)

