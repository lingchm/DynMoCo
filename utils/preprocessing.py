import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm 
import MDAnalysis as mda
from MDAnalysis.analysis import distances as ds

def get_average_trajectory_positions(u, sample_frequency = 1, warm_up_frames=50):
    
    atom_ids = u.select_atoms("protein and name CA").atoms.ids
    residue_indices = u.select_atoms("protein and name CA").atoms.resindices
    print("There are", len(atom_ids), "residues and", len(u.trajectory), "timestamps")
    atoms = u.select_atoms('protein').atoms
    
    u_avg_position = np.zeros(u.trajectory[0].positions[atom_ids].shape)
    for t in tqdm(range(warm_up_frames, len(u.trajectory), sample_frequency)): # len(u.trajectory)):
        u_avg_position += u.trajectory[t].positions[atom_ids]
            
    u_avg_position /= len(u.trajectory)
    
    return u_avg_position


def get_time_distance_matrix(u_list, method="ca", threshold=0, sample_frequency=100, warm_up_frames=0):
    """
    Input:
        u: a MDA Universe 
    Output:
        time_dist: a numpy array of shape (time, n_residue, n_residue)
    """

    if not isinstance(u_list, list):
        u_list = [u_list]
    
    atom_ids = u_list[0].select_atoms("protein and name CA").atoms.ids
    residue_indices = u_list[0].select_atoms("protein and name CA").atoms.resindices
    atoms = u_list[0].select_atoms('protein').atoms
    
    dists = []
    for u in u_list:
        print("There are", len(atom_ids), "residues and", len(u.trajectory), "timestamps")
        for t in range(warm_up_frames, len(u.trajectory), sample_frequency): # len(u.trajectory)):
            dist = np.zeros((len(residue_indices), len(residue_indices)))
            triu = np.triu_indices_from(dist, k=1)
            # compute distance based on CA atom of each residue 
            if method == "ca": 
                self_distances = ds.self_distance_array(u.trajectory[t].positions[atom_ids])
            # compute distance based on center of mass of each residue 
            else:
                self_distances = ds.self_distance_array(atoms.center_of_mass(compound='residues'))
            #dist[triu] = self_distances
            #dist.T[triu] = self_distances
            #dists.append(dist)
            dists.append(self_distances)

    time_dist = np.stack(dists)

    if threshold > 0:
        time_dist[np.where(time_dist<=threshold)] = 0

    del dists
    return time_dist 