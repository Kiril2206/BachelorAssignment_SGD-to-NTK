"""
Utilities for parameter flattening and unflattening.
Essential for converting JAX PyTrees into vectors for Jacobian computation.
"""

import jax
import jax.numpy as jnp
import numpy as np

def flatten_params(params_list):
    """
    Flattens a list of parameter tuples (W, b) into a single 1D vector.
    
    Args:
        params_list: The JAX parameter tree.
        
    Returns:
        flat_vector: 1D jnp.array of all parameters concatenated.
        treedef: Metadata to reconstruct the tree structure.
    """
    flat_params_leaves, treedef = jax.tree_util.tree_flatten(params_list)
    flat_params_leaves = [jnp.asarray(leaf) for leaf in flat_params_leaves]
    return jnp.concatenate([p.ravel() for p in flat_params_leaves]), treedef

def get_shapes_and_dtypes(params_list):
    """Extracts metadata (shape, dtype) for unflattening."""
    flat_params_leaves, _ = jax.tree_util.tree_flatten(params_list)
    return [(p.shape, p.dtype) for p in flat_params_leaves]

def unflatten_params(flat_params_vec, treedef, shapes_and_dtypes_meta):
    """
    Reconstructs the parameter tree from a flat 1D vector.
    
    Args:
        flat_params_vec: 1D array of parameters.
        treedef: Tree definition from flatten_params.
        shapes_and_dtypes_meta: Metadata from get_shapes_and_dtypes.
        
    Returns:
        The original parameter tree structure.
    """
    leaves = []
    current_pos = 0
    for shape, dtype in shapes_and_dtypes_meta:
        num_elements = np.prod(shape, dtype=int)
        leaf_flat = flat_params_vec[current_pos : current_pos + num_elements]
        leaves.append(leaf_flat.reshape(shape).astype(dtype))
        current_pos += num_elements
    return jax.tree_util.tree_unflatten(treedef, leaves)