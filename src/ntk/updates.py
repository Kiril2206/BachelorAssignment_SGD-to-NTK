import jax.numpy as jnp

def ntk_update_1_linearized(theta_k, J0, f_theta_k, y_train, learning_rate, N):
    """
    NTK 1: Discrete Gradient Descent with Linearized Jacobian.
    Theta_{k+1} = Theta_k - (eta/N) * J0.T * (f(Theta_k) - y)
    """
    residual = f_theta_k - y_train
    residual_flat = residual.ravel()
    update = (learning_rate / N) * jnp.dot(J0.T, residual_flat)
    return theta_k - update

def ntk_update_2_functional(theta_k, J0, Theta0, f_0, y_train, learning_rate, N, step_k):
    """
    NTK 2: Discrete Update with Functional Evolution.
    """
    error_0 = (f_0 - y_train).ravel()
    eigvals, eigvecs = jnp.linalg.eigh(Theta0)
    
    # Thesis Eq 16: Decay factor exp(-(2/N) * Theta0 * k)
    decay_scaler = jnp.exp(-(2.0 / N) * eigvals * step_k)
    
    decayed_error = jnp.dot(eigvecs, decay_scaler * jnp.dot(eigvecs.T, error_0))
    update = (learning_rate / N) * jnp.dot(J0.T, decayed_error)
    return theta_k - update

def ntk_update_3_integrated(theta_0, J0, Theta0, f_0, y_train, time_t, N):
    """
    NTK 3: Continuous-Time Integrated Parameter Evolution (One-Shot).
    Theta(t) = Theta_0 - J0.T * (I - exp(-Theta0 * t/N)) * Theta0_inv * (f_0 - y)
    """
    error_0 = (f_0 - y_train).ravel()
    eigvals, eigvecs = jnp.linalg.eigh(Theta0)
    
    # Safe inverse operator: (1 - exp(-lambda * t/N)) / lambda
    safe_eigvals = jnp.where(eigvals < 1e-10, 1.0, eigvals)
    operator_vals = (1.0 - jnp.exp(-(time_t / N) * eigvals)) / safe_eigvals
    operator_vals = jnp.where(eigvals < 1e-10, time_t/N, operator_vals)
    
    transformed_error = jnp.dot(eigvecs, operator_vals * jnp.dot(eigvecs.T, error_0))
    delta_theta = jnp.dot(J0.T, transformed_error)
    
    return theta_0 - delta_theta