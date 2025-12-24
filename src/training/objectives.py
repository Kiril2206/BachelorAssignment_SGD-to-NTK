"""
Generalized loss functions and factories.
"""
import jax.numpy as jnp
import jax

def mse_loss(params, x, y, model_forward):
    """Mean Squared Error for Regression."""
    preds = model_forward(params, x)
    # Ensure y is shaped correctly
    return jnp.mean((preds - y.reshape(preds.shape)) ** 2)

def cross_entropy_loss(params, x, y_one_hot, model_forward):
    """Cross Entropy for Classification."""
    logits = model_forward(params, x)
    # Log Softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits)
    # y_one_hot and log_probs must match shape
    return -jnp.mean(jnp.sum(y_one_hot * log_probs, axis=1))

def get_loss_fn(problem_type):
    """Factory to return the correct loss function."""
    if problem_type == 'regression':
        return mse_loss
    elif problem_type == 'classification':
        return cross_entropy_loss
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")