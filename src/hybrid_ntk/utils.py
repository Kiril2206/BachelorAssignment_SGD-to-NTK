def init_network_params(layer_dims, key):
    keys = jax.random.split(key, len(layer_dims) - 1)
    params = []
    for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        W = initializers.glorot_normal()(keys[i], (in_dim, out_dim))
        b = initializers.zeros(keys[i], (out_dim,))
        params.append((W, b))
    return params

@jit
def jax_relu(x):
    return jnp.maximum(0, x)

def jax_softmax(x, axis=-1):
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

def jax_forward(params, x):
    """Returns logits for classification."""
    activation = x
    for i, (W, b) in enumerate(params[:-1]):
        outputs = jnp.dot(activation, W) + b
        activation = jax_relu(outputs)
    final_W, final_b = params[-1]
    logits = jnp.dot(activation, final_W) + final_b
    return logits

def jax_update_params(params, grads, lr):
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]

@jit
def jax_compute_crossentropy_loss(params, x_batch, y_batch_one_hot):
    """Computes Cross-Entropy loss for classification."""
    logits = jax_forward(params, x_batch)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(y_batch_one_hot.astype(jnp.float32) * log_probs) / x_batch.shape[0]

jax_loss_grad_fn = jit(grad(jax_compute_crossentropy_loss, argnums=0))

def jax_predict_proba(params, x):
    return jax_softmax(jax_forward(params, x))

def compute_accuracy_jax(params, x, y_one_hot):
    """Computes accuracy for classification."""
    preds_proba = jax_predict_proba(params, x)
    return jnp.mean(jnp.argmax(preds_proba, axis=1) == jnp.argmax(y_one_hot, axis=1))

def flatten_params(params_list):
    flat_params_leaves, treedef = jax.tree_util.tree_flatten(params_list)
    flat_params_leaves = [jnp.asarray(leaf) for leaf in flat_params_leaves]
    return jnp.concatenate([p.ravel() for p in flat_params_leaves]), treedef

def unflatten_params(flat_params_vec, treedef, shapes_and_dtypes_meta):
    leaves = []
    current_pos = 0
    for shape, dtype in shapes_and_dtypes_meta:
        num_elements = np.prod(shape, dtype=int)
        leaves.append(jnp.asarray(flat_params_vec[current_pos: current_pos + num_elements], dtype=dtype).reshape(shape))
        current_pos += num_elements
    return jax.tree_util.tree_unflatten(treedef, leaves)

def get_shapes_and_dtypes(params_list):
    flat_params_meta, _ = jax.tree_util.tree_flatten(params_list)
    return [(p.shape, p.dtype) for p in flat_params_meta]

def compute_params_diff_norm(params1, params2):
    """Computes the Frobenius norm of the difference between two parameter lists."""
    diff_norms_sq = [
        jnp.sum((w1 - w2)**2) + jnp.sum((b1 - b2)**2)
        for (w1, b1), (w2, b2) in zip(params1, params2)
    ]
    return jnp.sqrt(jnp.sum(jnp.array(diff_norms_sq)))

def single_sample_forward_flat_params(flat_params_vec, single_x_input, treedef, shapes_and_dtypes_meta):
    unflattened_params_list = unflatten_params(flat_params_vec, treedef, shapes_and_dtypes_meta)
    return jax_forward(unflattened_params_list, single_x_input.reshape(1, -1))[0]

def matrix_exp_taylor(A, order=5):
    """Taylor expansion for matrix exponential: I + A + A^2/2! + ..."""
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for exponential.")

    result = jnp.eye(N, dtype=A.dtype)
    A_power_k = jnp.eye(N, dtype=A.dtype)
    factorial_k = 1.0

    for k in range(1, order + 1):
        A_power_k = jnp.dot(A_power_k, A)
        factorial_k *= k
        result += A_power_k / factorial_k
    return result

USE_JAX_EXPM = True # Set to False to use Taylor expansion

def compute_matrix_exp(A, taylor_order=5):
    if USE_JAX_EXPM:
        try:
            return jax.scipy.linalg.expm(A)
        except Exception as e:
            print(f"jax.scipy.linalg.expm failed: {e}. Falling back to Taylor expansion.")
            return matrix_exp_taylor(A, order=taylor_order)
    else:
        return matrix_exp_taylor(A, order=taylor_order)

@partial(jit, static_argnames=['num_classes_for_k0'])
def compute_empirical_ntk_k0(J_all_at_theta0, num_classes_for_k0):
    """Computes the empirical NTK for a multi-class output."""
    K0 = jnp.einsum('acp,bcp->ab', J_all_at_theta0, J_all_at_theta0)
    return K0

def evaluate_on_test_jax(params, X_test, Y_test_onehot):
    if params is None:
        print("\nTest Evaluation - No parameters provided.")
        return np.nan, np.nan
    
    test_loss = float(jax_compute_crossentropy_loss(params, X_test, Y_test_onehot))
    test_acc = float(compute_accuracy_jax(params, X_test, Y_test_onehot))

    print(f"\nTest Set Evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

def aggregate_histories(list_of_histories, metric_keys=['train_loss', 'val_loss', 'train_acc', 'val_acc']):
    if not list_of_histories:
        return {key: {'mean': [], 'std': []} for key in metric_keys}

    aggregated = {}
    for key in metric_keys:
        all_series = [h[key] for h in list_of_histories if key in h and h[key]]
        if not all_series:
            aggregated[key] = {'mean': [], 'std': []}
            continue

        max_len = max(len(s) for s in all_series)
        padded_series = []
        for s in all_series:
            padding = [np.nan] * (max_len - len(s))
            padded_series.append(s + padding)

        stacked_series = np.array(padded_series)

        means = []
        stds = []
        for i in range(max_len):
            column = stacked_series[:, i]
            valid_values = column[~np.isnan(column)]
            if valid_values.size > 0:
                means.append(np.mean(valid_values))
                stds.append(np.std(valid_values) if valid_values.size > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(np.nan)

        aggregated[key] = {
            'mean': means,
            'std': stds
        }
    return aggregated

def aggregate_scalar_metrics(list_of_scalar_metrics):
    if not list_of_scalar_metrics:
        return {'mean': np.nan, 'std': np.nan}
    metrics_array = np.array([m for m in list_of_scalar_metrics if m is not np.nan])
    if metrics_array.size == 0:
        return {'mean': np.nan, 'std': np.nan}
    return {
        'mean': np.nanmean(metrics_array),
        'std': np.nanstd(metrics_array)
    }
