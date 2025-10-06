import torch
import torch_geometric.nn as nn
from functools import partial


import numpy as np

def angle_between_vectors(u, v):
    """
    Calculate the angle between two 3D vectors in radians.

    Parameters:
    u (array-like): First vector (e.g., [x1, y1, z1]).
    v (array-like): Second vector (e.g., [x2, y2, z2]).

    Returns:
    float: Angle between the vectors in radians.
    """
    u = np.array(u)
    v = np.array(v)
    
    # Compute the dot product
    dot_product = np.dot(u, v)
    
    # Compute the magnitudes
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_u * norm_v)
    
    # To prevent numerical errors leading to out-of-bounds values for arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle in radians
    angle = np.arccos(cos_theta)
    
    return np.degrees(angle)


import numpy as np

def calculate_plane_angle(h1_1, h2_1, t_1, h1_2, h2_2, t_2):
    # Calculate normal vector for Plane 1
    v1_plane1 = np.array(h2_1) - np.array(h1_1)
    v2_plane1 = np.array(t_1) - np.array(h1_1)
    n1 = np.cross(v1_plane1, v2_plane1)
    
    # Calculate normal vector for Plane 2
    v1_plane2 = np.array(h2_2) - np.array(h1_2)
    v2_plane2 = np.array(t_2) - np.array(h1_2)
    n2 = np.cross(v1_plane2, v2_plane2)
    
    # Normalize the normal vectors
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    # Ensure the normals are not zero
    if n1_norm == 0 or n2_norm == 0:
        raise ValueError("One of the planes has zero normal vector, cannot compute angle.")
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    # Compute the angle between the planes
    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)  # Clip to handle numerical precision
    theta = np.arccos(cos_theta)  # Angle in radians
    
    # Calculate the intersection vector (cross product of the normal vectors)
    intersection_vector = np.cross(n1, n2)

    # Check if the planes are parallel
    if np.linalg.norm(intersection_vector) == 0:
        intersection_vector = None  # Planes are parallel or coincident

    return np.degrees(theta), intersection_vector # Convert to degrees for readability

