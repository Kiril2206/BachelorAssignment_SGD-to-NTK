"""
Multi-Layer Perceptron (MLP) implementation using JAX.
Generalized to support arbitrary depth and width via configuration.
"""

import jax
import jax.numpy as jnp
from jax.nn import initializers

def init_network_params(layer_dims, key):
    """
    Initialize MLP parameters with Glorot Normal initialization.
    
    Args:
        layer_dims (list[int]): Sequence of layer sizes [input, hidden..., output].
        key (jax.random.PRNGKey): Random seed for reproducibility.
        
    Returns:
        List]: List of (Weight, Bias) tuples.
    """
    keys = jax.random.split(key, len(layer_dims) - 1)
    params = []
    
    for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        # Thesis specifies Glorot Normal for weights and Zeros for biases
        W = initializers.glorot_normal()(keys[i], (in_dim, out_dim))
        b = initializers.zeros(keys[i], (out_dim,))
        params.append((W, b))
        
    return params

@jax.jit
def relu(x):
    """Rectified Linear Unit activation."""
    return jnp.maximum(0, x)

def forward(params, x):
    """
    Forward pass of the MLP.
    
    Args:
        params: List of (W, b) tuples.
        x: Input batch of shape (batch_size, input_dim).
        
    Returns:
        jnp.ndarray: Linear logits (no final activation). 
        Final activation (Softmax) is handled in the loss function.
    """
    activation = x
    # Iterate over all layers except the last
    for W, b in params[:-1]:
        outputs = jnp.dot(activation, W) + b
        activation = relu(outputs)
    
    # Final layer (Linear output)
    final_W, final_b = params[-1]
    final_outputs = jnp.dot(activation, final_W) + final_b
    return final_outputs