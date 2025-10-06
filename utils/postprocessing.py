import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from collections import defaultdict

import pandas as pd

def chain_annotation(nodes, u):
    segment_freq = defaultdict(int)
    for n in nodes:
        seg = u.select_atoms("protein and name CA")[n].segid
        segment_freq[seg] += 1
    for k, v in segment_freq.items():
        segment_freq[k] /= len(nodes)
    sorted_items = sorted(segment_freq.items(), key=lambda x: x[1], reverse=True)
    formatted_str = ", ".join([f"{k} ({v*100:.0f}%)" for k, v in sorted_items]) 
    return formatted_str

def domain_annotation(nodes, domain_mapping):
    domains = {}
    for n in nodes:
        if n in domain_mapping:
            dom = domain_mapping[n] 
        else:
            if n < 1000: dom = "loopA"
            else: dom = "loopB"
        if dom not in domains:
            domains[dom] = 1
        else:
            domains[dom] += 1
    for k, v in domains.items():
        domains[k] /= len(nodes)
    
    sorted_items = sorted(domains.items(), key=lambda x: x[1], reverse=True)
    formatted_str = ", ".join([f"{k} ({v*100:.0f}%)" for k, v in sorted_items]) 
    return formatted_str

def format_residue_ranges(residues):
    """
    Format a list of residue integers into condensed string format.
    E.g., [1,2,3,10,14,16,17,18,19,20] → '1-3, 10, 14, 16-20'
    """
    if not residues:
        return ""

    residues = sorted(set(residues))
    ranges = []
    start = prev = residues[0]

    for res in residues[1:]:
        if res == prev + 1:
            prev = res
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = prev = res

    # Add the last range
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ', '.join(ranges)

# Function to create a sorting key based on the presence of words in the correct order
def sort_key(text):
    word_order = ['beta-propeller (100%)', 'beta-propeller', 'thigh (100%)', 'thigh', 'loopA', 'calf1 (100%)', 'calf1', 
              'calf2', 'calf2 (100%)', 
              'betaI (100%)', 'betaI', 'hybrid (100%)', 'hybrid', 'psi', 'loopB', 'egf1 (100%)', 'egf1', 
              'egf2 (100%)', 'egf2', 'egf3 (100%)', 'egf3', 'egf4 (100%)', 'egf4', 'betaTD (100%)', 'betaTD']
    # Initialize a list to store the index of the words found in the text
    key = []
    for word in word_order:
        # if word == text[:len(word)]:
        if word in text:
            key.append(word_order.index(word))  # Add the index of the word in the desired order
        else:
            key.append(len(word_order))  # If the word is not found, assign it a very high value
    return key

def compute_displacement_and_rotation(X, Y):
    """
    Compute the displacement of the center and Euler angles aligning X to Y.
    
    Parameters:
    - X: (N1, 3) array of positions at time t1
    - Y: (N2, 3) array of positions at time t2
    
    Returns:
    - displacement_vector: 1D array (3,), the vector from centroid of X to centroid of Y
    - euler_angles: 1D array (3,), rotation angles (in radians) around x, y, z axes
    """
    assert X.shape == Y.shape, "X and Y must have the same shape"

    # Compute centroids
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)

    # Displacement vector of the average position
    displacement_vector = centroid_Y - centroid_X

    # Center the point clouds
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Compute covariance matrix
    H = X_centered.T @ Y_centered

    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R_matrix = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T

    # Convert rotation matrix to Euler angles (in radians)
    r = R_scipy.from_matrix(R_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)  # Use degrees=True for degrees

    return displacement_vector, euler_angles

def compute_displacement_and_rotation(X, Y):
    """
    Compute the displacement of the center and Euler angles aligning X to Y.
    
    Parameters:
    - X: (N1, 3) array of positions at time t1
    - Y: (N2, 3) array of positions at time t2
    
    Returns:
    - displacement_vector: 1D array (3,), the vector from centroid of X to centroid of Y
    - euler_angles: 1D array (3,), rotation angles (in radians) around x, y, z axes
    """
    assert X.shape == Y.shape, "X and Y must have the same shape"

    # Compute centroids
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)

    # Displacement vector of the average position
    displacement_vector = centroid_Y - centroid_X

    # Center the point clouds
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # Compute covariance matrix
    H = X_centered.T @ Y_centered

    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R_matrix = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T

    # Convert rotation matrix to Euler angles (in radians)
    r = R_scipy.from_matrix(R_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)  # Use degrees=True for degrees

    return displacement_vector, euler_angles



def post_process(communities, u, resindices_to_index, residue_to_domain):
    
    rep = 0
    extensions = list(u[rep].keys())
    n_segments = len(extensions)
    
    node_dict = defaultdict(lambda: defaultdict(str))
    for community, nodes in communities.items():
        if community in communities.keys():
            for node, timesteps in nodes.items():
                for t in timesteps:
                    node_dict[node][t] = community
                    
    communities_snapshot = {}
    for c in communities.keys():
        communities_snapshot[c] = {}
        for t in range(n_segments):
            communities_snapshot[c][t] = []
        for n in list(communities[c].keys()):
            for t in communities[c][n]:
                communities_snapshot[c][t].append(n)

    t1, t2 = 0, n_segments - 1

    df_results = pd.DataFrame()
    print("Num communities:", len(communities_snapshot.keys()))

    # compute displacement
    for c in list(communities_snapshot.keys()):
        
        df_results.loc[c, "ID"] = "C" + c
        nodes_t1 = communities_snapshot[c][t1] # resindex 
        nodes_t2 = communities_snapshot[c][t2]
        overlap = list(set(nodes_t1) & set(nodes_t2))
        only_in_list1 = len(set(nodes_t1) - set(nodes_t2))
        only_in_list2 = len(set(nodes_t2) - set(nodes_t1))
        
        cur_coms = defaultdict(int)
        for n in nodes_t1:
            cur_com = node_dict[n][t]
            cur_coms["C" + cur_com] += 1
        for k, v in cur_coms.items():
            cur_coms[k] /= len(nodes_t1)

        df_results.loc[c, "Chain"] = chain_annotation([resindices_to_index[resindex] for resindex in nodes_t1], u[rep][extensions[0]])
        df_results.loc[c, "Domain"] = domain_annotation(nodes_t1, residue_to_domain)
        resids = [
            u[rep][extensions[t1]].select_atoms("protein and name CA")[resindices_to_index[resindex]].resid for resindex in nodes_t1
        ]
        df_results.loc[c, "Residues"] = format_residue_ranges(resids)
        df_results.loc[c, "N"] = len(nodes_t1)

    df_results.set_index("ID", inplace=True)

    # Apply the function to create a sorting key and sort the DataFrame
    df_results['sort_key'] = df_results['Domain'].apply(sort_key)
    df_sorted = df_results.sort_values('sort_key').drop(columns='sort_key')
    # note that some commnities appear at a later point but are empty at t0

    new_index = ["C" + str(i+1) for i in range(df_sorted.shape[0])]
    index_mapping = dict(zip(df_sorted.index, new_index))
    # print(index_mapping)
    df_sorted.index = new_index
    # df_sorted = df_sorted[df_sorted["N"] > 0] # some communities are empty at t0 

    communities_sorted = {index_mapping["C" + k][1:]: v for k, v in communities.items()}

    communities_included = sorted([k for k, v in communities_sorted.items()])
    print("Num communities at t0:", df_sorted[df_sorted["N"] > 0].shape[0])
    print("Num communities included:", len(communities_included))
    
    node_dict = defaultdict(lambda: defaultdict(str))
    for community, nodes in communities_sorted.items():
        for node, timesteps in nodes.items():
            for t in timesteps:
                node_dict[node][t] = community
                    
    communities_snapshot = {}
    for c in communities_sorted.keys():
        communities_snapshot[c] = {}
        for t in range(len(extensions)):
            communities_snapshot[c][t] = []
        for n in list(communities_sorted[c].keys()):
            for t in communities_sorted[c][n]:
                communities_snapshot[c][t].append(n)
                
    len(communities_snapshot.keys())
    
    # characterization
    t1, t2 = 0, n_segments - 1
    split_threshold = 0.1

    # compute displacement
    for com in list(df_sorted.index):

        c = com[1:]
        nodes_t1 = communities_snapshot[c][t1] # resindex
        nodes_t2 = communities_snapshot[c][t2]
        # nodes_t1 = [resindices_to_index[i] for i in communities_snapshot[c][t1]]
        # nodes_t2 = [resindices_to_index[i] for i in communities_snapshot[c][t2]]
        overlap = list(set(nodes_t1) & set(nodes_t2))
        only_in_list1 = len(set(nodes_t1) - set(nodes_t2))
        only_in_list2 = len(set(nodes_t2) - set(nodes_t1))
        overlap_index = [resindices_to_index[resindex] for resindex in overlap]
        nodes_t1_index = [resindices_to_index[resindex] for resindex in nodes_t1]
        nodes_t2_index = [resindices_to_index[resindex] for resindex in nodes_t2]
        
        cur_coms = defaultdict(list)
        for n in nodes_t1: # for each node in this community at current time 
            cur_com = node_dict[n][t2] # position of node at later time 
            if len(cur_com) > 0:
                cur_coms["C" + cur_com].append(n)  

        df_sorted.loc[com, "N2"] = len(nodes_t1)
        df_sorted.loc[com, "unchanged"] = len(overlap)
        df_sorted.loc[com, "unchanged_pct"] = np.round(len(overlap) / (len(nodes_t1) + 1e-6), 3)
        df_sorted.loc[com, "added"] = only_in_list2
        df_sorted.loc[com, "added_pct"] = np.round(only_in_list2 / (len(nodes_t2) + 1e-6), 2)
        df_sorted.loc[com, "removed"] = only_in_list1
        df_sorted.loc[com, "removed_pct"] = np.round(only_in_list1 / (len(nodes_t1) + 1e-6), 2)
        nodes_to  = []
        for key, value in cur_coms.items():
            if key[1:] == c: continue
            resids = [
                u[rep][extensions[t1]].select_atoms("protein and name CA")[resindices_to_index[resindex]].resid for resindex in value
            ]
            nodes_to.append(f"{key}: {resids}".replace("[","").replace("]",""))
        nodes_to = "\n".join(nodes_to)
        df_sorted.loc[com, "nodes_to"] = nodes_to
        # df_sorted.loc[com, "nodes_to"] = str.join(', ', list(cur_coms.keys()))
        
        resids = [u[rep][extensions[t1]].select_atoms("protein and name CA")[resindices_to_index[resindex]].resid for resindex in nodes_t1]
        df_sorted.loc[com, "Change"] = "-"
        
        if len(nodes_t2) < 0.2 * len(nodes_t1): # more than 80% of nodes are moved 
            merge_destination = [k for k, v in cur_coms.items()]
            df_sorted.loc[com, "Change"] = f"Merge -> {com}, " + str.join(", ", merge_destination)
            for to in merge_destination:
                if to in df_sorted.index:
                    df_sorted.loc[to, "Change"] = f"Merge -> {to}, {com}"
            
        elif df_sorted.loc[com, "removed_pct"] > split_threshold:
            sorted_items = sorted(cur_coms.items(), key=lambda x: x[1], reverse=True)
            splits = [(k, v) for k, v in sorted_items]
            splits = [k for k, v in sorted_items]
            if len(splits) > 1:
                # df_sorted.loc[com, "Change"] = "Split -> " + ", ".join([f"{k} ({v*100:.0f}%)" for k, v in splits]) 
                df_sorted.loc[com, "Change"] = "Split -> " + ", ".join([f"{k}" for k in splits]) 

        df_sorted.loc[com, "Change"] = "Stable" if df_sorted.loc[com, "Change"] == "-" else df_sorted.loc[com, "Change"]

        disp_string, rot_string = "", ""

        if df_sorted.loc[com, "Change"][:6] == "Stable":
            positions_t1 = u[rep][extensions[t1]].select_atoms("protein and name CA").positions[overlap_index]
            positions_t2 = u[rep][extensions[t2]].select_atoms("protein and name CA").positions[overlap_index]
            disp, rotation = compute_displacement_and_rotation(positions_t1, positions_t2)
            disp_distance = np.linalg.norm(disp)
            disp_string += str(round(disp_distance, 1)) + " (" + str.join(",", np.round(disp,1).astype(str)) + ")"
            rot_string += "(" + str.join(",", np.round(rotation,1).astype(str)) + ")"
            
        elif df_sorted.loc[com, "Change"][:5] == "Merge":
            positions_t1 = u[rep][extensions[t1]].select_atoms("protein and name CA").positions[nodes_t1_index]
            positions_t2 = u[rep][extensions[t2]].select_atoms("protein and name CA").positions[nodes_t1_index]
            disp, rotation = compute_displacement_and_rotation(positions_t1, positions_t2)
            disp_distance = np.linalg.norm(disp)
            disp_string += str(round(disp_distance, 1)) + " (" + str.join(",", np.round(disp,1).astype(str)) + ")"
            rot_string += "(" + str.join(",", np.round(rotation,1).astype(str)) + ")"
        
        else:
            for item in sorted_items:
                c_, nodes = item[0][1:], item[1]
                # print("Split: C" + c_, nodes)
                overlap = list(set(communities_snapshot[c_][t1]) & set(communities_snapshot[c_][t2]))
                overlap_index = [resindices_to_index[resindex] for resindex in overlap]
                if len(disp_string) > 0:
                    disp_string += "\n"
                    rot_string += "\n"
                positions_t1 = u[rep][extensions[t1]].select_atoms("protein and name CA").positions[overlap_index]
                positions_t2 = u[rep][extensions[t2]].select_atoms("protein and name CA").positions[overlap_index]
                disp, rotation = compute_displacement_and_rotation(positions_t1, positions_t2)
                disp_distance = np.linalg.norm(disp)
                disp_string += str(round(disp_distance, 1)) + " (" + str.join(",", np.round(disp,1).astype(str)) + ")"
                rot_string += "(" + str.join(",", np.round(rotation,1).astype(str)) + ")"
        
        df_sorted.loc[com, "Disp (A)"] = disp_string
        df_sorted.loc[com, "RotAx"] = rot_string
        
    
    node_dict = defaultdict(lambda: defaultdict(str))
    for community, nodes in communities_sorted.items():
        if community in communities_included:
            for node, timesteps in nodes.items():
                for t in timesteps:
                    node_dict[node][t] = community
                    
    communities_snapshot = {}
    for c in communities_included:
        communities_snapshot[c] = {}
        for t in range(len(extensions)):
            communities_snapshot[c][t] = []
        for n in list(communities_sorted[c].keys()):
            for t in communities_sorted[c][n]:
                communities_snapshot[c][t].append(n)
                
    len(communities_snapshot.keys())
    
    return df_sorted, communities_sorted
