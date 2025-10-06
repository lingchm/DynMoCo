import torch
import networkx as nx
import numpy as np
import scipy.sparse as sparse
from torch.utils.data import Dataset
from torch_sparse import SparseTensor
import torch

# Assuming graph_utils contains the necessary functions for normalization and conversion
from utils.graph_utils import convert_scipy_sparse_to_sparse_tensor, normalize_graph # Custom utility functions

class DynamicGraphDataset(Dataset):
    def __init__(self, graph_list, comembership, device='cpu'):
        """
        Args:
            graph_list (list of networkx graphs): A list of graphs at each time step.
            extensions (list): A list to handle the time steps or any additional data.
            device (str): The device to store the tensors (default is 'cuda').
        """
        self.graph_list = graph_list
        self.device = device
        
        self.features_list = []
        self.adjacency_list = []
        self.norm_graph_list = []
        self.graph_list_tensor = []
        self.comembership = comembership
        
        # Process the data (adjacency matrices and features)
        self._process_data()

    def _process_data(self):
        """
        Process the data: Convert graph to adjacency matrices, extract features, 
        and normalize the graphs.
        """
        for s in range(len(self.graph_list)):
            self.features_list.append([])
            self.adjacency_list.append([])
            self.norm_graph_list.append([])
            self.graph_list_tensor.append([])
                
            for t in range(len(self.graph_list[s])):
                # Load and process the data
                adjacency = sparse.csr_matrix(nx.to_scipy_sparse_array(self.graph_list[s][t], format='csr'))
                features = torch.FloatTensor(np.array([self.graph_list[s][t].nodes[n]['feature'] for n in sorted(self.graph_list[s][t].nodes)]))
                graph_ = convert_scipy_sparse_to_sparse_tensor(adjacency).to(self.device)
                graph_normalized = convert_scipy_sparse_to_sparse_tensor(
                    normalize_graph(adjacency.copy())
                ).to(self.device)
                
                if t == 0:
                    print(f"Data shape: {features.shape}, adjacency shape: {adjacency.shape}")

                self.adjacency_list[s].append(adjacency)
                self.features_list[s].append(features)
                self.graph_list_tensor[s].append(graph_)
                self.norm_graph_list[s].append(graph_normalized)
                
        self.feature_shape = features.shape[1]

    def __len__(self):
        """
        Returns the number of time steps (sequences).
        """
        return len(self.graph_list)

    def __getitem__(self, idx):
        """
        Fetches a single item (graph sequence) from the dataset.

        Args:
            idx (int): Index of the sequence.

        Returns:
            dict: A dictionary containing 'graphs', 'norm_graphs', and 'features'.
        """
        # Return the graph (adjacency) tensors and corresponding features
        sample = {
            'features_list': self.features_list[idx],  # Node features
            'adjacency_list': self.adjacency_list[idx], # Original graph tensor
            'graph_list': self.graph_list_tensor[idx],  # Original graph tensor
            'norm_graphs_list': self.norm_graph_list[idx], # Normalized graph tensor
            'comembership': self.comembership  # Comembership matrices
        }

        return sample
    

def collate_sparse_graph(batch):
    """
    Collate function to handle SparseTensor batching.

    Args:
        batch (list of dict): List of data samples returned by __getitem__.

    Returns:
        dict: A dictionary with batched tensors.
    """
    # Initialize lists for storing batched graphs, features, etc.
    graph_list = []
    norm_graph_list = []
    adjacency_list = []
    features_list = []
    comembership_list = []

    # Iterate over each sample in the batch
    for sample in batch:
        graph_list.append(sample['graph_list'])
        adjacency_list.append(sample['adjacency_list'])
        norm_graph_list.append(sample['norm_graphs_list'])
        features_list.append(sample['features_list'])
        comembership_list.append(sample['comembership'])

    # Convert lists to tensors, if needed, or handle batching of SparseTensors.
    # For SparseTensor, we will leave them as lists of SparseTensors.
    return {
        'graph_list': graph_list,
        'adjacency_list': adjacency_list,
        'norm_graphs_list': norm_graph_list,
        'features_list': features_list,
        'comembership': comembership_list
    }



class GraphNodeDataset(Dataset):
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.node_ids = list(graph.nodes)
        self.features = torch.stack([
            torch.tensor(graph.nodes[node]['feature'], dtype=torch.float32)
            for node in self.node_ids
        ])
        # self.adjacency = torch.tensor(
        #     nx.to_numpy_array(graph, nodelist=self.node_ids, dtype=float),
        #     dtype=torch.float32
        # )
    
    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        return self.features[idx]