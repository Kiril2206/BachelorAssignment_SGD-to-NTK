"""
Core logic for computing Neural Tangent Kernel dynamics.
"""

import jax
import jax.numpy as jnp
from jax import jacfwd, vmap
from src.models.mlp import forward
from src.utils.flattening import unflatten_params

def get_forward_flat(treedef, shapes):
    def forward_flat(flat_params, x):
        params = unflatten_params(flat_params, treedef, shapes)
        return forward(params, x.reshape(1, -1)).squeeze()
    return forward_flat

def compute_jacobian(flat_params_0, x_train, treedef, shapes):
    """
    Computes the Jacobian J0 of the network output w.r.t parameters.
    """
    forward_fn = get_forward_flat(treedef, shapes)

    # Compute Jacobian per sample using vmap
    J_per_sample = vmap(jacfwd(forward_fn, argnums=0), (None, 0))(flat_params_0, x_train)
    
    # FIX: Use  to get the integer size from the shape tuple
    # shape is (N_samples, N_outputs, N_params) -> reshape to (N_samples*N_outputs, N_params)
    J_reshaped = J_per_sample.reshape(-1, flat_params_0.shape[0])
    return J_reshaped

def compute_ntk_matrix(J0):
    return jnp.dot(J0, J0.T)