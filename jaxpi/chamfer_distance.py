import numpy as np
import jax
from jax import jit
import jax.numpy as jnp

def chamfer_distance(set1, set2, bi_direction = True):


    """Compute Chamfer distance between two sets of points."""
    def pairwise_distance_matrix(set1, set2):
        """Compute pairwise distances between points in two sets."""
        # Compute squared distances
        set1_norm = jnp.sum(set1**2, axis=1, keepdims=True)
        set2_norm = jnp.sum(set2**2, axis=1, keepdims=True)
        distances = set1_norm + set2_norm.T - 2 * jnp.dot(set1, set2.T)
        return distances 
    
    pairwise_distances = pairwise_distance_matrix(set1, set2)
    chamfer_dist = jnp.mean( jnp.min(pairwise_distances, axis = 0))
    chamfer_dist += jnp.mean( jnp.min(pairwise_distances, axis = 1))
    # if bi_direction:
    #     # Compute pairwise distances between sets
    #     distances_set2_to_set1 = pairwise_distance_matrix(set2, set1)

    #     # Compute Chamfer distance
    #     chamfer_dist = jnp.mean(jnp.min(distances_set1_to_set2, axis=1)) + \
    #                 jnp.mean(jnp.min(distances_set2_to_set1, axis=1))
    # else:
    #     chamfer_dist = jnp.mean(jnp.min(distances_set1_to_set2, axis=1)) 
    return chamfer_dist

def chamfer_distance_directional(set1, set2):


    """Compute Chamfer distance between two sets of points."""
    def pairwise_distance_matrix(set1, set2):
        """Compute pairwise distances between points in two sets."""
        # Compute squared distances
        set1_norm = jnp.sum(set1**2, axis=1, keepdims=True)
        set2_norm = jnp.sum(set2**2, axis=1, keepdims=True)
        distances = set1_norm + set2_norm.T - 2 * jnp.dot(set1, set2.T)
        return distances 
    
    pairwise_distances = pairwise_distance_matrix(set1, set2)
    chamfer_dist = jnp.mean( jnp.min(pairwise_distances, axis = 1))
    return chamfer_dist


chamfer_distance_jit = jit(chamfer_distance)
chamfer_distance_directional_jit = jit(chamfer_distance_directional)

# chamfer_distance_jit = chamfer_distance

# Example usage:
set1 = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
set2 = jnp.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

# Compute Chamfer distance
distance = chamfer_distance_jit(set1, set2)
print("Chamfer Distance:", distance)
