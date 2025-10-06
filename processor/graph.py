from utils.preprocessing import get_average_trajectory_positions, get_time_distance_matrix
import tnetwork as tn
import numpy as np
import pandas as pd
import torch
import networkx as nx
from tqdm import tqdm
from scipy import stats
import scipy.sparse as sp
from typing import Dict, List, Tuple
from collections import defaultdict

def build_comembership_matrix(domain_to_residues, resindices_to_index):
    # Determine total number of nodes if not provided
    num_nodes=len(resindices_to_index.keys())
    print("Number of nodes:", num_nodes)
    comembership = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for resindices in domain_to_residues.values():
        nodes = [resindices_to_index[resindex] for resindex in resindices if resindex in resindices_to_index.keys()]
        nodes = torch.tensor(nodes)
        if nodes.shape[0] > 1:
           comembership[nodes.unsqueeze(0), nodes.unsqueeze(1)] = 1  # M[i, j] = 1 if i, j in same domain

    return comembership

# def build_dynamic_graph(u, extensions, rep, config, domain_to_residues, residue_to_domain, 
#                         warm_up_frames = 1, bound_thd = 5, sample_frequency = 1, pval_thd = 1e-5, node_attributes = "coords"):

#     domains = list(domain_to_residues.keys())
#     T = len(extensions)
#     snapshots = list(range(T))

#     atom_ids = u[rep][extensions[0]].select_atoms("protein and name CA").atoms.ids
#     # residue_indices = u[rep][extensions[0]].select_atoms("protein and name CA").atoms.resindices

#     dist_matrices = {} 
#     graph = tn.DynGraphSN(frequency=1)
#     graph_sequence = []

#     # add nodes
#     residue_indices = []
#     index_to_residue = {}
#     resindices_to_index = {}
    
#     i, j = 0, 0
#     for resindex in u[rep][extensions[0]].select_atoms("protein and name CA").atoms.resindices:
#         index_to_residue[i] = resindex
#         i += 1
#         if resindex not in residue_to_domain:
#             # print(f"resindex {resindex} not in residue_to_domain")
#             continue 
#         if "transmembrane" in residue_to_domain[resindex] or "cytoplasmic" in residue_to_domain[resindex]:
#             continue 
#         residue_indices.append(resindex)
#         resindices_to_index[resindex] = j
#         j += 1
#     residue_indices = np.array(residue_indices)
#     # print("residue_indices:", residue_indices.shape)
#     graph.add_nodes_presence_from(residue_indices, snapshots) #add nodes a,b,c in snapshots 2 to 5

#     from collections import defaultdict
        
#     # add edges
#     for tt, ext in enumerate(extensions):
#         print("ext:", ext)
#         avg_positions = get_average_trajectory_positions(u[rep][ext], warm_up_frames=warm_up_frames)
#         if node_attributes == "deviation" or node_attributes == "both":
#             if tt > 0:
#                 prior_positions = get_average_trajectory_positions(u[rep][extensions[tt-1]], warm_up_frames=warm_up_frames)
#                 deviation = avg_positions - prior_positions
#             else:
#                 deviation = avg_positions * 0
#         time_dist = get_time_distance_matrix([u[rep][ext] for rep in range(config['n_replications'])] , method="ca", sample_frequency=sample_frequency, warm_up_frames=warm_up_frames)

#         # CMD significantly bonded
#         candidates_idx = np.where(np.mean(time_dist, axis=0) < bound_thd * 1.6)[0]
#         bonded_pval = np.ones(time_dist.shape[1])
#         # pval_thd = 0.05 / candidates_idx.shape[0] if bonferroni else 0.05
#         for idx in tqdm(candidates_idx):
#             distances = time_dist[:,idx]
#             t_statistic, p_value = stats.ttest_1samp(distances, popmean=bound_thd, alternative="less")
#             if p_value <= pval_thd: 
#                 bonded_pval[idx] = p_value 
                
#         bonded = bonded_pval < pval_thd
#         # print(f"There are {np.sum(bonded)} bonds in {ext}")
        
#         n = u[rep][extensions[0]].select_atoms("protein and name CA").atoms.resindices.shape[0]
#         dist = np.zeros((n, n))
#         triu = np.triu_indices_from(dist, k=1)
#         dist[triu] = bonded
#         dist.T[triu] = bonded
#         dist_matrices[ext] = dist
        
#         edges = [] 
#         df_domain_edges = pd.DataFrame(np.zeros((len(domains), len(domains))), columns=domains, index=domains)
#         for i in range(dist.shape[0]):
#             if index_to_residue[i] not in residue_indices: continue
                
#             for j in range(i+1, dist.shape[1]):
#                 if index_to_residue[j] not in residue_indices: continue
                
#                 # non-covalent distance below or covalent 
#                 if dist[i,j] > 0 or abs(atom_ids[i] - atom_ids[j]) == 1:
#                     edges.append((index_to_residue[i], index_to_residue[j]))
#                     df_domain_edges.loc[residue_to_domain[index_to_residue[i]], residue_to_domain[index_to_residue[j]]] += 1
                
#         graph.add_interactions_from(edges, snapshots[tt]) #link a and b in snapshot 2
#         # print("There are", len(edges), "edges in extension", ext)
        
#         del time_dist, bonded_pval, bonded, avg_positions, edges
        
#     return graph 


def contruct_graph_dygraph(u, extensions, config, residue_to_domain, 
                           warm_up_frames = 1, node_attributes="coords", bound_thd = 5, 
                           bonferroni = True, pval_thd = 1e-5):

    T = len(extensions)
    snapshots = list(range(T))

    graph_sequences = []
    dygraph_sequences = [] 

    for rep in range(config['n_replications']):
        atom_ids = u[rep][extensions[0]].select_atoms("protein and name CA").resids
        segids = u[rep][extensions[0]].select_atoms("protein and name CA").segids
        # residue_indices = u[rep][extensions[0]].select_atoms("protein and name CA").atoms.resindices

        dist_matrices = {} 
        graph = tn.DynGraphSN(frequency=1)
        graph_sequence = []

        # add nodes
        residue_indices = []
        index_to_residue = {}
        resindices_to_index = {}
        
        i, j = 0, 0
        for resindex in u[rep][extensions[0]].select_atoms("protein and name CA").atoms.resindices:
            index_to_residue[i] = resindex
            i += 1
            if resindex not in residue_to_domain:
                # print(f"resindex {resindex} not in residue_to_domain")
                continue 
            if "transmembrane" in residue_to_domain[resindex] or "cytoplasmic" in residue_to_domain[resindex]:
                continue 
            residue_indices.append(resindex)
            resindices_to_index[resindex] = j
            j += 1
            
        residue_indices = np.array(residue_indices)
        # print("residue_indices:", residue_indices.shape)
        graph.add_nodes_presence_from(residue_indices, snapshots) #add nodes a,b,c in snapshots 2 to 5

        # add edges
        for tt, ext in enumerate(extensions):
            print("ext:", ext)
            avg_positions = get_average_trajectory_positions(u[rep][ext], warm_up_frames=warm_up_frames)
            if node_attributes == "deviation" or node_attributes == "both":
                if tt > 0:
                    prior_positions = get_average_trajectory_positions(u[rep][extensions[tt-1]], warm_up_frames=warm_up_frames)
                    deviation = avg_positions - prior_positions
                else:
                    deviation = avg_positions * 0
            time_dist = get_time_distance_matrix([u[rep][ext]] , method="ca", sample_frequency=config['sample_frequency'], warm_up_frames=warm_up_frames)

            # CMD significantly bonded
            candidates_idx = np.where(np.mean(time_dist, axis=0) < bound_thd * 1.6)[0]
            bonded_pval = np.ones(time_dist.shape[1])
            # pval_thd = 0.05 / candidates_idx.shape[0] if bonferroni else 0.05
            for idx in candidates_idx:
                distances = time_dist[:,idx]
                t_statistic, p_value = stats.ttest_1samp(distances, popmean=bound_thd, alternative="less")
                if p_value <= pval_thd: 
                    bonded_pval[idx] = p_value 
                    
            bonded = bonded_pval < pval_thd
            # print(f"    There are {np.sum(bonded)} bonds")
            
            n = u[rep][extensions[0]].select_atoms("protein and name CA").atoms.resindices.shape[0]
            dist = np.zeros((n, n))
            triu = np.triu_indices_from(dist, k=1)
            dist[triu] = bonded
            dist.T[triu] = bonded
            dist_matrices[ext] = dist
            
            edges = [] 
            count_covalent = 0
            for i in range(dist.shape[0]):
                if index_to_residue[i] not in residue_indices: continue
                    
                for j in range(i+1, dist.shape[1]):
                    if index_to_residue[j] not in residue_indices: continue
                    
                    # covalent 
                    if abs(atom_ids[i] - atom_ids[j]) == 1 and segids[i] == segids[j]:
                        edges.append((index_to_residue[i], index_to_residue[j]))
                        count_covalent += 1
                    
                    # non-covalent distance below or covalent 
                    elif dist[i,j] > 0:
                        edges.append((index_to_residue[i], index_to_residue[j]))
                        
            edges = list(set(edges))
            
            graph.add_interactions_from(edges, snapshots[tt]) #link a and b in snapshot 2
            # print("     There are", len(edges), "edges")
            # print("     There are", count_covalent, "covalent edges")
            
            # count node degree
            degree_dict = defaultdict(int)
            for e1, e2 in edges:
                degree_dict[e1] += 1
                degree_dict[e2] += 1  # undirected graph
            
            # add nodes with features to the graph list 
            G = nx.Graph()
            for i, feat in enumerate(avg_positions):
                
                if index_to_residue[i] not in residue_indices: continue
                
                if node_attributes == "constant":
                    G.add_node(index_to_residue[i], feature=torch.ones(feat.shape[0]))
                    
                elif node_attributes == "coords":
                    G.add_node(index_to_residue[i], feature=feat)
                    
                elif node_attributes == "identity":
                    identity_feat = torch.zeros(n)
                    identity_feat[i] = 1.0
                    G.add_node(index_to_residue[i], feature=identity_feat)

                elif node_attributes == "degree":
                    deg = degree_dict[index_to_residue[i]]
                    G.add_node(index_to_residue[i], feature=torch.tensor([deg], dtype=torch.float))
                
                elif node_attributes == "deviation":
                    G.add_node(index_to_residue[i], feature=torch.tensor(deviation[i], dtype=torch.float))
                    
                elif node_attributes == "both":
                    G.add_node(index_to_residue[i], feature=torch.tensor([feat, deviation[i]], dtype=torch.float).flatten())
                else:
                    print("WARNING: node_attributes is not defined")
                    
            G.add_edges_from(edges)
            
            graph_sequence.append(G)
            
            del time_dist, bonded_pval, bonded, degree_dict, avg_positions, edges
        
        graph_sequences.append(graph_sequence)
        dygraph_sequences.append(graph)
            
        graph.summary()
    
    return graph_sequences, dygraph_sequences, resindices_to_index, dist_matrices

# -----------------------------------------------------------------------------
# Helpers and adapters to match MFC-TopoReg data loader outputs
# -----------------------------------------------------------------------------

def sparse_to_tuple(sparse_mx: sp.coo_matrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Convert a scipy.sparse COO matrix into tuple format (coords, values, shape).

    This mirrors the helper used in the original repository's dataloader so that
    downstream code expecting that structure continues to work unchanged.
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def md_graphs_to_mfc_format(
    graph_sequence: List[nx.Graph],
    residue_to_domain: Dict[int, str],
    device: torch.device,
) -> Tuple[List[Tuple[sp.coo_matrix, torch.Tensor, List[int]]], int]:
    """Convert a sequence of MD-derived NetworkX graphs into MFC snapshot format.

    Returns data in the exact structure expected by the original repository's
    `load_graphs` → `NetworkSnapshots`, i.e. a list of triples per snapshot:
      - adj: scipy.sparse.coo_matrix adjacency
      - features: torch.sparse.FloatTensor identity features (on provided device)
      - labels: list[int] node labels aligned with the node order used for adj

    The number of clusters (n_cluster) is computed as the number of distinct
    domain labels present across all nodes in the provided graphs.
    """
    # Collect the domain set actually present in the provided graphs
    domain_set = set()
    for G in graph_sequence:
        for node in G.nodes():
            if node in residue_to_domain:
                domain_set.add(residue_to_domain[node])

    # Stable ordering of domains for reproducibility
    domain_list = sorted(domain_set)
    domain_to_id: Dict[str, int] = {d: i for i, d in enumerate(domain_list)}
    n_cluster = len(domain_to_id)

    snapshots: List[Tuple[sp.coo_matrix, torch.Tensor, List[int]]] = []
    for G in graph_sequence:
        # Ensure consistent node ordering across adj, features, and labels
        nodes_order: List[int] = list(G.nodes())

        # Build adjacency in scipy COO using the same node order
        adj_coo: sp.coo_matrix = sp.coo_matrix(nx.adjacency_matrix(G, nodelist=nodes_order))

        # One-hot identity features in sparse torch format (device-aware)
        num_nodes = adj_coo.shape[0]
        eye_sp = sp.coo_matrix(np.eye(num_nodes), dtype=np.float32)
        coords, values, shape = sparse_to_tuple(eye_sp)
        features = torch.sparse.FloatTensor(
            torch.LongTensor(coords.T),
            torch.FloatTensor(values),
            torch.Size(shape),
        ).to(device)

        # Map node → domain id; unseen nodes default to -1 (filtered later if any)
        labels: List[int] = []
        for n in nodes_order:
            dom = residue_to_domain.get(n, None)
            if dom is None:
                # If any node is out of mapping, assign a temporary label bucket
                # to keep type compatibility without failing the pipeline.
                labels.append(-1)
            else:
                labels.append(domain_to_id[dom])

        snapshots.append((adj_coo, features, labels))

    return snapshots, n_cluster
