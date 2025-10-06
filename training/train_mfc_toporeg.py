import os 
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm 
import matplotlib.cm as cm
from scipy import stats
from collections import defaultdict
from time import time 
from typing import List
import torch
import json

import MDAnalysis as mda
import plotly.graph_objs as go
import tnetwork as tn
import networkx as nx
import random

from utils.visualization import plot_molecules, plot_molecule
from utils.preprocessing import get_average_trajectory_positions, get_time_distance_matrix

##
import processor.data as data_processor
import processor.graph as graph_processor
import processor.model as model_processor

from processor.filtration import WrcfLayer
from processor.trainer import base_train, retrain_with_topo
from processor.filtration import build_community_graph

import processor.metrics as pr_metrics

###################################
# pip install gudhi
# pip install pot
# pip install eagerpy
###################################

##########
class Args(dict):
    def __init__(self, n_cluster, file_name=None, network_type="MFC") -> None:
        self.encoded_space_dim = 50
        self.n_cluster = n_cluster  # clusters
        self.num_epoch = 701 #1000
        self.learning_rate = 0.001 # for topo
        self.LAMBDA = 1
        self.card = 20 # num of ph considered
        self.file_name = file_name
        self.network_type = network_type
        self.start_mf = 500


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda") # "cpu"

data_integrin = {
    # "zhenhai_smd": ["a5b1", "aVb3"],
    # "zhenhai_clamp": ["a5b1", "aVb3"],
    # "martin": ["alpha2bbeta3", "alphaVbeta3"],
    "pd1_l2": ["PD-1"]
}

rep = 0

for dataset_name in data_integrin.keys():
    for integrin in data_integrin[dataset_name]:
        print(f"Running {dataset_name} {integrin}")

        # load data and create graph
        # TODO: check sample_frequency of zhenhai_smd. a5b1 -> 2, aVb3 -> 10, correct?
        u, extensions, config = data_processor.load_data(dataset_name, integrin)
        domain_to_residues, residue_to_domain, domain_to_chain = graph_processor.return_domain_residue_chain(u, integrin, extensions)

        graph_sequences, dygraph_sequences = graph_processor.contruct_graph_dygraph(
            u=u,
            extensions=extensions,
            config=config,
            domain_to_residues=domain_to_residues,
            residue_to_domain=residue_to_domain
        )

        # define model
        model_init = model_processor.InitModel(device=device)
        # graph_sequences[rep] : list (nx.Graph for each timestep)
        # residue_to_domain : dict, {node_id: domain_name}


        # TODO: check if this is correct
        snapshot_list, n_cluster = graph_processor.md_graphs_to_mfc_format(
            graph_sequence=graph_sequences[rep],
            residue_to_domain=residue_to_domain,
            device=device
        )
        print(len(snapshot_list))   # num of snapshots (time step)
        print(n_cluster)        # num of labels (clusters)

        # check first snapshot
        adj, features, labels = snapshot_list[0]
        print(adj.shape)        # adjacency size
        print(features.shape)   # feature size
        print(len(labels))      # num of nodes

        ### RUN
        args = Args(n_cluster, file_name=None, network_type="MFC") # fix 20 cluster or assume known n_cluster
        model_list = []
        dgm_list = []
        wrcf_layer_dim0 = WrcfLayer(dim=0, card=args.card)
        wrcf_layer_dim1 = WrcfLayer(dim=1, card=args.card)

        results_raw = [] 
        results_topo = []

        # base deep clustering training
        for idx, (adj,features,labels) in enumerate(snapshot_list):
            print(f"Base training: {idx} ===================")
            
            model = model_init(adj, features.size(1), args)
            model_list.append(model)
            
            base_train(model, features, adj, args, str(idx))
            with torch.no_grad():
                # if network_type == "SDCN":
                #     _, Q, _, Z = model(features,adj)
                # else:
                _, Z, Q = model(features, adj)
                results_raw.append([
                    Z.cpu().detach().numpy(),
                    Q.cpu().detach().numpy(),
                    adj,labels
                ])
                # record dgm at each time step
                community_graph = build_community_graph(Q, adj)
                dgm0 = wrcf_layer_dim0(community_graph)
                dgm1 = wrcf_layer_dim1(community_graph)
                dgm_list.append([dgm0,dgm1])

        # topological regulaized training
        for t in range(len(snapshot_list)):
            m = model_list[t]
            adj,features,labels = snapshot_list[t]
            if t == 0:
                gt_dgm = [None, dgm_list[t+1]]
            elif t == len(snapshot_list)-1: 
                gt_dgm = [dgm_list[t-1], None]
            else:
                gt_dgm = [dgm_list[t-1],dgm_list[t+1]]
            retrain_with_topo(m, gt_dgm, adj, features, args, str(t))
            with torch.no_grad():
                # if network_type == "SDCN":
                #     _, Q, _, Z = m(features,adj)
                # else:
                _, Z, Q = m(features,adj)
                results_topo.append([
                    Z.cpu().detach().numpy(),
                    Q.cpu().detach().numpy(),
                    adj,labels
                ])
                # update dgm at time 
                community_graph = build_community_graph(Q,adj)
                dgm0_new = wrcf_layer_dim0(community_graph)
                dgm1_new = wrcf_layer_dim1(community_graph)
                dgm_list[t] = [dgm0_new,dgm1_new]

        # ================= Metrics (Modularity & Conductance) =================
        def _hard_clusters_from_Q(Q_np: np.ndarray) -> np.ndarray:
            return np.argmax(Q_np, axis=1)

        def _compute_metrics_time_series(results):
            modularity_ts = []
            conductance_ts = []
            modularity_gt_ts = []
            conductance_gt_ts = []
            for Z_np, Q_np, adj_sp, labels in results:
                clusters_pred = _hard_clusters_from_Q(Q_np)
                labels_np = np.asarray(labels)
                # adjacency is scipy.sparse (coo); metrics.* expects sparse
                mod_pred = pr_metrics.modularity(adj_sp, clusters_pred)
                cond_pred = pr_metrics.conductance(adj_sp, clusters_pred)
                modularity_ts.append(float(mod_pred))
                conductance_ts.append(float(cond_pred))
                # also compute against provided labels for reference
                mod_gt = pr_metrics.modularity(adj_sp, labels_np)
                cond_gt = pr_metrics.conductance(adj_sp, labels_np)
                modularity_gt_ts.append(float(mod_gt))
                conductance_gt_ts.append(float(cond_gt))
            return {
                "modularity": modularity_ts,
                "conductance": conductance_ts,
                "modularity_gt": modularity_gt_ts,
                "conductance_gt": conductance_gt_ts,
                "modularity_mean": float(np.mean(modularity_ts)),
                "conductance_mean": float(np.mean(conductance_ts)),
                "modularity_gt_mean": float(np.mean(modularity_gt_ts)),
                "conductance_gt_mean": float(np.mean(conductance_gt_ts)),
            }

        metrics_base = _compute_metrics_time_series(results_raw)
        metrics_topo = _compute_metrics_time_series(results_topo)

        print("Metrics (base):", json.dumps({
            "modularity_mean": metrics_base["modularity_mean"],
            "conductance_mean": metrics_base["conductance_mean"],
        }, indent=2))
        print("Metrics (topo):", json.dumps({
            "modularity_mean": metrics_topo["modularity_mean"],
            "conductance_mean": metrics_topo["conductance_mean"],
        }, indent=2))

        # save results
        os.makedirs("toporeg_results", exist_ok=True)
        with open(os.path.join("toporeg_results", f"metrics_{dataset_name}_{integrin}.json"), "w") as f:
            json.dump({
                "base": metrics_base,
                "topo": metrics_topo,
            }, f, indent=2)
