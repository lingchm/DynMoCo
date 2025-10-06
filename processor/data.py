import os
import MDAnalysis as mda
import numpy as np


DATA_ROOT = "data" 


def load_data(dataset_type, integrin):
    
    if dataset_type == 'clamp':
        assert integrin in ["a5b1", "aVb3"], f"Integrin {integrin} with clamp not implemented"
        return load_clamp(integrin)
    elif dataset_type == 'ramp':
        assert integrin in ["alphaVbeta3", "alpha2bbeta3"], f"Integrin {integrin} with ramp not implemented"
        return load_ramp(integrin)
    else:
        raise NotImplementedError(f"Dataset type {dataset_type} not implemented")


def load_clamp(integrin):
    data_folder = os.path.join(DATA_ROOT, integrin + " Clamp")
    if integrin == "a5b1":
        extensions = ["12nm", "14nm", "16nm", "18nm"]
        n_replications = 6 
        sample_frequency = 2
    else:
        extensions = ["3nm", "11nm", "16nm", "18nm"]
        n_replications = 5
        sample_frequency = 10
        
    size = 4
    bound_thd = 5

    u = {rep: None for rep in range(n_replications)}
    for rep in u:
        u[rep] = {}
        
        for ext in extensions:
            
            if integrin != "aVb3":
                topology = f"{data_folder}/{ext}/{integrin}.md.nw.{rep}.gro"
                trajectory = f"{data_folder}/{ext}/{integrin}.md.nw.{rep}.xtc"
            else:
                topology = f"{data_folder}/{ext}/integrin.md.nowater.{str(rep+1)}.gro"
                trajectory = f"{data_folder}/{ext}/integrin.md.nowater.{str(rep+1)}.xtc"

            u[rep][ext] = mda.Universe(topology, trajectory) 
            
            # assign segment IDs
            segment_a = u[rep][ext].add_Segment(segid='A')
            segment_b = u[rep][ext].add_Segment(segid='B')
            max_resid = np.max(u[rep][ext].select_atoms("protein").residues.resnums)
            max_resid_index = np.where(u[rep][ext].select_atoms("protein").residues.resnums == max_resid)[0][0]
            for i in range(u[rep][ext].select_atoms("protein").residues.resnums.shape[0]):
                if i < max_resid_index + 1: 
                    u[rep][ext].select_atoms("protein").residues[i].segment = segment_a
                else: 
                    u[rep][ext].select_atoms("protein").residues[i].segment = segment_b
            u[rep][ext].add_TopologyAttr("chainID")
            segment_a.atoms.chainIDs = "A"
            segment_b.atoms.chainIDs = "B"

    config = {
        'dataset_name': f'{integrin}_clamp',
        'integrin': integrin,
        'data_folder': data_folder,
        'dataset_type': "clamp",
        'size': size,
        'bound_thd': bound_thd,
        'n_replications': n_replications,
        'sample_frequency': sample_frequency
    }

    return u, extensions, config


def load_ramp(integrin):
    assert integrin in ["alpha2bbeta3", "alphaVbeta3"]
    data_folder = os.path.join(DATA_ROOT, integrin + " Ramp")
    size = 4
    bound_thd = 5
    frames = 10
    segments = 4
    sample_frequency = 1
    extensions = ["Extension", "Bending"]

    if integrin != "alphaVbeta3": 
        n_replications = 3
    else: 
        n_replications = 2
        
    u = {}

    # process the data into segments 
    for rep in range(1, n_replications + 1):
        u[rep-1] = {}
        for extension in extensions:
            topology = f"{data_folder}/{integrin}.pdb"
            trajectory = f"{data_folder}/{extension}_Trajectories/Replica{str(rep)}.xtc"
            u[rep-1][extension] = mda.Universe(topology, trajectory)  
            if rep == 1:
                T = len(u[rep-1][extension].trajectory)
                starts = list(range(0, T - frames, (T - frames) // (segments - 1)))
            for s in starts:
                e = s + frames
                with mda.Writer(f"{data_folder}/{extension}_Trajectories/Replica{str(rep)}_{s}-{e}.dcd", u[rep-1][extension].atoms.n_atoms) as W:
                    for ts in u[rep-1][extension].trajectory[s:e]:
                        W.write(u[rep-1][extension].atoms)    
    
    extensions = [str(i)+"-"+str(i+frames) for i in starts]
    
    # load the data into segments 
    setting = "Extension"
    u = {rep: None for rep in range(n_replications)}
    for rep in u:
        u[rep] = {}
        
        for ext in extensions:
            
            topology = f"{data_folder}/{integrin}.pdb"
            trajectory = f"{data_folder}/{setting}_Trajectories/Replica{str(rep+1)}_{ext}.dcd"
            u[rep][ext] = mda.Universe(topology, trajectory) # 
        
        topology = f"{data_folder}/{integrin}.pdb"
        trajectory = f"{data_folder}/{setting}_Trajectories/Replica{str(rep+1)}.xtc"
        u[rep]["all"] = mda.Universe(topology, trajectory)

    max_resid_index = u[0][ext].select_atoms(f"name CA and segid A").resindices[-1]

    config = {
        'dataset_name': f'{integrin}_ramp',
        'integrin': integrin,
        'data_folder': data_folder,
        'dataset_type': "ramp",
        'size': size,
        'bound_thd': bound_thd,
        'frames': frames,
        'segments': segments,
        'n_replications': n_replications,
        'sample_frequency': sample_frequency,
        'max_resid_index': max_resid_index
    }
    
    return u, extensions, config

