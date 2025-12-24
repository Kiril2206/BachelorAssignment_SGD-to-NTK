"""
Heuristics for detecting the 'Lazy Training' regime onset.
"""

import jax.numpy as jnp

def check_parameter_stability(current_params, old_params, threshold):
    """
    Heuristic 1: Parameter Norm Difference.
    Checks ||theta_t - theta_{t-k}||_F < epsilon.
    """
    diff_sq_sum = 0.0
    for (w1, b1), (w2, b2) in zip(current_params, old_params):
        diff_sq_sum += jnp.sum((w1 - w2)**2) + jnp.sum((b1 - b2)**2)
    
    norm_diff = jnp.sqrt(diff_sq_sum)
    return norm_diff < threshold, norm_diff

def check_ntk_stability(current_ntk, initial_ntk, old_ntk_diff, threshold):
    """
    Heuristic 2: Empirical NTK Stability.
    Checks the rate of change of the NTK matrix deviation.
    Delta Theta_t = |||Theta_t - Theta_0||_F - ||Theta_{t-k} - Theta_0||_F| < delta
    """
    current_deviation = jnp.linalg.norm(current_ntk - initial_ntk)
    
    # Calculate the rate of change of the deviation
    rate_of_change = jnp.abs(current_deviation - old_ntk_diff)
    
    return rate_of_change < threshold, current_deviation