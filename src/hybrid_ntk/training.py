def run_sgd_epochs(params_initial, X_train_sgd, Y_train_onehot_sgd, X_val_full, Y_val_onehot_full,
                   start_epoch_idx, num_epochs_to_run, # start_epoch_idx is 0-based
                   batch_size, lr_sgd, key_sgd_loop, phase_label="SGD"):
    print(f"\n--- Starting {phase_label} Training Phase (Epochs {start_epoch_idx + 1} to {start_epoch_idx + num_epochs_to_run}) ---")
    params = params_initial # Start from provided parameters
    N_train_sgd = X_train_sgd.shape[0]
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    sgd_phase_start_time = time.time()
    current_batch_size = min(batch_size, N_train_sgd)

    for epoch_offset in range(num_epochs_to_run):
        actual_epoch_num_display = start_epoch_idx + epoch_offset + 1 # For printing (1-based)

        key_sgd_loop, subkey_perm = jax.random.split(key_sgd_loop)
        indices = jax.random.permutation(subkey_perm, N_train_sgd)
        
        # --- Mini-batch update loop ---
        for i in range(0, N_train_sgd, current_batch_size):
            X_batch = X_train_sgd[indices[i:i + current_batch_size]]
            Y_batch = Y_train_onehot_sgd[indices[i:i + current_batch_size]]
            grads = jax_loss_grad_fn(params, X_batch, Y_batch)
            params = jax_update_params(params, grads, lr_sgd)
            
        # --- Full-dataset metric logging (end of epoch) ---
        train_loss = float(jax_compute_crossentropy_loss(params, X_train_sgd, Y_train_onehot_sgd))
        train_acc = float(compute_accuracy_jax(params, X_train_sgd, Y_train_onehot_sgd))
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss = float(jax_compute_crossentropy_loss(params, X_val_full, Y_val_onehot_full))
        val_acc = float(compute_accuracy_jax(params, X_val_full, Y_val_onehot_full))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"{phase_label} Epoch {actual_epoch_num_display} - Train L: {train_loss:.4f}, A: {train_acc*100:.2f}% | Val L: {val_loss:.4f}, A: {val_acc*100:.2f}%")

    sgd_phase_time = time.time() - sgd_phase_start_time
    print(f"{phase_label} phase ({num_epochs_to_run} epochs) took {sgd_phase_time:.2f} seconds.")
    return params, history, sgd_phase_time

def run_sgd_monitoring_switch(
    params_initial, X_train_sgd, Y_train_onehot_sgd, X_val_full, Y_val_onehot_full,
    max_sgd_epochs, batch_size, lr_sgd, key_sgd_loop,
    switch_config,
    X_ntk_monitor_subset, num_classes
):
    print(f"\n--- Starting SGD Phase (Monitoring for Switch using '{switch_config['method']}') ---")
    params = params_initial
    N_train_sgd = X_train_sgd.shape[0]
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    sgd_phase_start_time = time.time()
    current_batch_size = min(batch_size, N_train_sgd)
    
    # --- Switch condition specific setup ---
    method = switch_config['method']
    if method == 'param_norm':
        k = switch_config.get('param_norm_window', 3)
        param_history = deque(maxlen=k + 1)
        param_history.append(copy.deepcopy(params))
        
    elif method == 'ntk_norm':
        k = switch_config.get('ntk_norm_window', 3)
        ntk_total_diff_history = deque(maxlen=k + 1)
        
        _, treedef = flatten_params(params)
        shapes_meta = get_shapes_and_dtypes(params)
        partial_apply_fn = partial(single_sample_forward_flat_params, treedef=treedef, shapes_and_dtypes_meta=shapes_meta)
        jac_fn_single = jax.jacrev(partial_apply_fn, argnums=0)
        J_vmap_fn = jit(jax.vmap(lambda p, x: jac_fn_single(p, x), in_axes=(None, 0), out_axes=0))
        params_flat_initial, _ = flatten_params(params_initial)
        J_initial = J_vmap_fn(params_flat_initial, X_ntk_monitor_subset)
        K_initial = compute_empirical_ntk_k0(J_initial, num_classes)
        ntk_total_diff_history.append(0.0)
        
    epoch_at_switch = max_sgd_epochs

    for epoch in range(max_sgd_epochs):
        actual_epoch_num_display = epoch + 1
        epoch_start_time = time.time()
        key_sgd_loop, subkey_perm = jax.random.split(key_sgd_loop)
        indices = jax.random.permutation(subkey_perm, N_train_sgd)
        
        for i in range(0, N_train_sgd, current_batch_size):
            X_batch = X_train_sgd[indices[i:i + current_batch_size]]
            Y_batch = Y_train_onehot_sgd[indices[i:i + current_batch_size]]
            grads = jax_loss_grad_fn(params, X_batch, Y_batch)
            params = jax_update_params(params, grads, lr_sgd)
            
        # Log metrics
        train_loss = float(jax_compute_crossentropy_loss(params, X_train_sgd, Y_train_onehot_sgd))
        train_acc = float(compute_accuracy_jax(params, X_train_sgd, Y_train_onehot_sgd))
        val_loss = float(jax_compute_crossentropy_loss(params, X_val_full, Y_val_onehot_full))
        val_acc = float(compute_accuracy_jax(params, X_val_full, Y_val_onehot_full))
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"SGD (Monitoring) Epoch {actual_epoch_num_display} - Train L: {train_loss:.4f}, A: {train_acc*100:.2f}% | Val L: {val_loss:.4f}, A: {val_acc*100:.2f}%", end='')
        
        # --- Check switch condition ---
        switch_now = False
        diff_norm = np.nan
        
        if method == 'fixed_epoch':
            if actual_epoch_num_display >= switch_config['fixed_switch_epoch']:
                switch_now = True
        elif method == 'param_norm':
            k = switch_config.get('param_norm_window', 3)
            param_history.append(copy.deepcopy(params))
            if len(param_history) == k + 1:
                diff_norm = compute_params_diff_norm(param_history[-1], param_history[0])
                print(f" | Param diff norm (k={k}) = {diff_norm:.6f}", end='')
                if diff_norm < switch_config['param_norm_threshold']:
                    switch_now = True
        
        elif method == 'ntk_norm':
            k = switch_config.get('ntk_norm_window', 3)
            params_flat_current, _ = flatten_params(params)
            J_current = J_vmap_fn(params_flat_current, X_ntk_monitor_subset)
            K_current = compute_empirical_ntk_k0(J_current, num_classes)
            ntk_diff_total = float(jnp.linalg.norm(K_current - K_initial, 'fro'))
            ntk_total_diff_history.append(ntk_diff_total)
            
            if len(ntk_total_diff_history) == k + 1:
                diff_norm = abs(ntk_total_diff_history[-1] - ntk_total_diff_history[0])
                print(f" | NTK stability (k={k}) = {diff_norm:.4f}", end='')
                if diff_norm < switch_config['ntk_norm_threshold']:
                    switch_now = True

        print(f" (took {time.time() - epoch_start_time:.2f}s)")
        if switch_now:
            epoch_at_switch = actual_epoch_num_display
            print(f">>> Switching condition '{method}' met at epoch {epoch_at_switch} over a {k}-epoch window. <<<")
            break
            
    if epoch == max_sgd_epochs - 1 and not switch_now:
        epoch_at_switch = max_sgd_epochs
        print(f">>> Max SGD epochs ({max_sgd_epochs}) reached without meeting switch condition. This run will be flagged. <<<")
    
    sgd_phase_time = time.time() - sgd_phase_start_time
    print(f"SGD monitoring phase ({epoch_at_switch} epochs) took {sgd_phase_time:.2f} seconds.")
    
    final_history = {k: v[:epoch_at_switch] for k, v in history.items()}
    return params, final_history, sgd_phase_time, epoch_at_switch

def run_sgd_scouting(
    params_initial, X_train_sgd, Y_train_onehot_sgd, X_val_full, Y_val_onehot_full,
    scouting_epochs, batch_size, lr_sgd, key_sgd_loop,
    switch_method, # 'param_norm' or 'ntk_norm'
    X_ntk_monitor_subset, num_classes,
    param_norm_window, ntk_norm_window
):
    print(f"\n--- Starting SGD Scouting Run for {scouting_epochs} Epochs (Method: {switch_method}) ---")
    params = params_initial
    N_train_sgd = X_train_sgd.shape[0]
    history = {'val_loss': [], 'val_acc': [], 'norm_diff': []}
    scouting_start_time = time.time()
    current_batch_size = min(batch_size, N_train_sgd)

    # --- Monitoring setup based on the chosen method ---
    if switch_method == 'param_norm':
        k = param_norm_window
        param_history = deque(maxlen=k + 1)
        param_history.append(copy.deepcopy(params))

    elif switch_method == 'ntk_norm':
        k = ntk_norm_window
        ntk_total_diff_history = deque(maxlen=k + 1)
        
        _, treedef = flatten_params(params)
        shapes_meta = get_shapes_and_dtypes(params)
        partial_apply_fn = partial(single_sample_forward_flat_params, treedef=treedef, shapes_and_dtypes_meta=shapes_meta)
        jac_fn_single = jax.jacrev(partial_apply_fn, argnums=0)
        J_vmap_fn = jit(jax.vmap(lambda p, x: jac_fn_single(p, x), in_axes=(None, 0), out_axes=0))
        params_flat_initial, _ = flatten_params(params_initial)
        J_initial = J_vmap_fn(params_flat_initial, X_ntk_monitor_subset)
        K_initial = compute_empirical_ntk_k0(J_initial, num_classes)
        ntk_total_diff_history.append(0.0)
    
    for epoch in range(scouting_epochs):
        key_sgd_loop, subkey_perm = jax.random.split(key_sgd_loop)
        indices = jax.random.permutation(subkey_perm, N_train_sgd)
        
        for i in range(0, N_train_sgd, current_batch_size):
            X_batch = X_train_sgd[indices[i:i + current_batch_size]]
            Y_batch = Y_train_onehot_sgd[indices[i:i + current_batch_size]]
            grads = jax_loss_grad_fn(params, X_batch, Y_batch)
            params = jax_update_params(params, grads, lr_sgd)

        # Log performance metrics
        history['val_loss'].append(float(jax_compute_crossentropy_loss(params, X_val_full, Y_val_onehot_full)))
        val_acc = float(compute_accuracy_jax(params, X_val_full, Y_val_onehot_full))
        history['val_acc'].append(val_acc)
        
        norm_diff = np.nan
        # --- Unified calculation logic ---
        if switch_method == 'param_norm':
            k = param_norm_window
            param_history.append(copy.deepcopy(params))
            if len(param_history) == k + 1:
                 norm_diff = compute_params_diff_norm(param_history[-1], param_history[0])

        elif switch_method == 'ntk_norm':
            k = ntk_norm_window
            params_flat_current, _ = flatten_params(params)
            J_current = J_vmap_fn(params_flat_current, X_ntk_monitor_subset)
            K_current = compute_empirical_ntk_k0(J_current, num_classes)
            ntk_diff_total = float(jnp.linalg.norm(K_current - K_initial, 'fro'))
            
            ntk_total_diff_history.append(ntk_diff_total)
            if len(ntk_total_diff_history) == k + 1:
                norm_diff = abs(ntk_total_diff_history[-1] - ntk_total_diff_history[0])
        
        history['norm_diff'].append(float(norm_diff))
        
        print(f"Scouting Epoch {epoch + 1}/{scouting_epochs} - Val Acc: {val_acc*100:.2f}% | Norm Diff (k={k}): {norm_diff:.4f}")

    print(f"Scouting run took {time.time() - scouting_start_time:.2f} seconds.")
    return history

def run_ntk1_phase(params_sgd, X_train_ntk, Y_train_onehot_ntk, X_val_full, Y_val_onehot_full,
                   ntk_epochs, batch_size, lr_ntk, key_ntk_loop):
    print("\n--- Starting NTK 1 Phase ---")
    N_train_ntk = X_train_ntk.shape[0]
    
    theta_0_params_unflat = params_sgd
    theta_0_flat, treedef_0 = flatten_params(theta_0_params_unflat)
    shapes_meta_0 = get_shapes_and_dtypes(theta_0_params_unflat)
    theta_k_flat = jnp.copy(theta_0_flat)

    partial_apply_fn = partial(single_sample_forward_flat_params, treedef=treedef_0, shapes_and_dtypes_meta=shapes_meta_0)
    jac_fn_single_sample = jax.jacrev(partial_apply_fn, argnums=0)
    J_at_theta0_single_sample = jit(lambda single_x: jac_fn_single_sample(theta_0_flat, single_x))
    J_batch_at_theta0_vmap = jit(jax.vmap(J_at_theta0_single_sample, in_axes=(0), out_axes=0))
    predict_batch_theta_k_vmap = jit(
        jax.vmap(partial(single_sample_forward_flat_params, treedef=treedef_0, shapes_and_dtypes_meta=shapes_meta_0),
                 in_axes=(None, 0), out_axes=0))

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    ntk1_start_time = time.time()

    for ntk_iter in range(ntk_epochs):
        key_ntk_loop, subkey_perm = jax.random.split(key_ntk_loop)
        indices = jax.random.permutation(subkey_perm, N_train_ntk)
        total_param_update_contrib = jnp.zeros_like(theta_k_flat)
        for i in range(0, N_train_ntk, batch_size):
            X_batch = X_train_ntk[indices[i:i + batch_size]]
            Y_batch_onehot = Y_train_onehot_ntk[indices[i:i + batch_size]]
            J_b_at_theta0 = J_batch_at_theta0_vmap(X_batch)
            logits_b_at_thetak = predict_batch_theta_k_vmap(theta_k_flat, X_batch)
            pred_probas_b_at_thetak = jax_softmax(logits_b_at_thetak)
            Error_batch = pred_probas_b_at_thetak - Y_batch_onehot
            batch_contrib = jnp.einsum('bcp,bc->p', J_b_at_theta0, Error_batch)
            total_param_update_contrib += batch_contrib

        effective_lr_ntk = (2.0 * lr_ntk) / N_train_ntk
        theta_k_flat -= effective_lr_ntk * total_param_update_contrib

        current_params_unflat = unflatten_params(theta_k_flat, treedef_0, shapes_meta_0)
        train_loss = float(jax_compute_crossentropy_loss(current_params_unflat, X_train_ntk, Y_train_onehot_ntk))
        train_acc = float(compute_accuracy_jax(current_params_unflat, X_train_ntk, Y_train_onehot_ntk))
        val_loss = float(jax_compute_crossentropy_loss(current_params_unflat, X_val_full, Y_val_onehot_full))
        val_acc = float(compute_accuracy_jax(current_params_unflat, X_val_full, Y_val_onehot_full))
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"NTK 1 Iter {ntk_iter + 1}/{ntk_epochs} - Train (full) L: {train_loss:.4f}, A: {train_acc*100:.2f}% | Val (full) L: {val_loss:.4f}, A: {val_acc*100:.2f}%")

    ntk1_time = time.time() - ntk1_start_time
    print(f"NTK 1 phase ({ntk_epochs} iterations) took {ntk1_time:.2f} seconds.")
    final_params_ntk1 = unflatten_params(theta_k_flat, treedef_0, shapes_meta_0)
    return final_params_ntk1, history, ntk1_time

def run_ntk2_phase(params_sgd, X_train_ntk, Y_train_onehot_ntk, X_val_full, Y_val_onehot_full, sgd_epochs,
                   ntk_epochs, lr_ntk, taylor_order_ntk2):
    print("\n--- Starting NTK 2 Phase ---")
    N_train_ntk = X_train_ntk.shape[0]
    num_classes = Y_train_onehot_ntk.shape[1]

    theta_0_params_unflat = params_sgd
    theta_0_flat, treedef_0 = flatten_params(theta_0_params_unflat)
    shapes_meta_0 = get_shapes_and_dtypes(theta_0_params_unflat)
    theta_k_flat = jnp.copy(theta_0_flat)

    # --- Precompute items related to ?_0 ---
    print("NTK2: Computing J(X_train_ntk, Theta_0) and K_0...")
    partial_apply_fn_theta0 = partial(single_sample_forward_flat_params, treedef=treedef_0, shapes_and_dtypes_meta=shapes_meta_0)
    jac_fn_single_sample_theta0 = jax.jacrev(partial_apply_fn_theta0, argnums=0)
    J_at_theta0_single_sample = jit(lambda single_x: jac_fn_single_sample_theta0(theta_0_flat, single_x))
    J_all_at_theta0_vmap = jit(jax.vmap(J_at_theta0_single_sample, in_axes=(0), out_axes=0))
    J_all_at_theta0 = J_all_at_theta0_vmap(X_train_ntk) # (N, C, P)

    K0 = compute_empirical_ntk_k0(J_all_at_theta0, num_classes) # (N, N)

    predict_batch_theta0_vmap = jit(jax.vmap(partial(single_sample_forward_flat_params, treedef=treedef_0, shapes_and_dtypes_meta=shapes_meta_0),
                                             in_axes=(None, 0), out_axes=0))
    logits_all_at_theta0 = predict_batch_theta0_vmap(theta_0_flat, X_train_ntk)
    pred_probas_all_at_theta0 = jax_softmax(logits_all_at_theta0)
    Error_all_at_theta0 = pred_probas_all_at_theta0 - Y_train_onehot_ntk # (N, C)
    print("NTK2: J(X_train_ntk, Theta_0), K_0, and Error_0 computed.")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    ntk2_start_time = time.time()

    for k in range(sgd_epochs + 1, sgd_epochs + ntk_epochs + 1):
        matrix_in_exp = -(2/N_train_ntk) * K0 * (k - 3)
        exp_term_matrix = compute_matrix_exp(matrix_in_exp, taylor_order_ntk2) # (N, N)
        damped_error_contribution_k = jnp.dot(exp_term_matrix, Error_all_at_theta0) # (N,N) @ (N,C) -> (N,C)
        total_param_update_contrib = jnp.einsum('ncp,nc->p', J_all_at_theta0, damped_error_contribution_k)

        theta_k_flat -= lr_ntk * total_param_update_contrib

        current_params_unflat = unflatten_params(theta_k_flat, treedef_0, shapes_meta_0)
        train_loss = float(jax_compute_crossentropy_loss(current_params_unflat, X_train_ntk, Y_train_onehot_ntk))
        train_acc = float(compute_accuracy_jax(current_params_unflat, X_train_ntk, Y_train_onehot_ntk))
        val_loss = float(jax_compute_crossentropy_loss(current_params_unflat, X_val_full, Y_val_onehot_full))
        val_acc = float(compute_accuracy_jax(current_params_unflat, X_val_full, Y_val_onehot_full))
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        print(f"NTK 2 (Matrix Exp) Iter {k}/{ntk_epochs+sgd_epochs} - Train L: {train_loss:.4f}, A: {train_acc*100:.2f}% | Val L: {val_loss:.4f}, A: {val_acc*100:.2f}%")

    ntk2_time = time.time() - ntk2_start_time
    print(f"NTK 2 phase (Matrix Exp, {ntk_epochs} iterations) took {ntk2_time:.2f} seconds.")
    final_params_ntk2 = unflatten_params(theta_k_flat, treedef_0, shapes_meta_0)
    return final_params_ntk2, history, ntk2_time

def run_ntk3_phase(params_sgd, X_train_ntk, Y_train_onehot_ntk, X_val_full, Y_val_onehot_full,
                   lr_ntk, lambda_ntk3_reg, T_factor_ntk3, taylor_order_ntk3):
    print(f"\n--- Starting NTK 3 Phase (Matrix Exp, T_factor={T_factor_ntk3}) ---")
    N_train_ntk = X_train_ntk.shape[0]
    num_classes = Y_train_onehot_ntk.shape[1]
    
    theta_0_params_unflat = params_sgd
    theta_0_flat, treedef_0 = flatten_params(theta_0_params_unflat)
    shapes_meta_0 = get_shapes_and_dtypes(theta_0_params_unflat)

    print("NTK3: Computing J(X_train_ntk, Theta_0), K_0, and Error_0...")
    partial_apply_fn_theta0 = partial(single_sample_forward_flat_params, treedef=treedef_0, shapes_and_dtypes_meta=shapes_meta_0)
    jac_fn_single_sample_theta0 = jax.jacrev(partial_apply_fn_theta0, argnums=0)
    J_at_theta0_single_sample = jit(lambda single_x: jac_fn_single_sample_theta0(theta_0_flat, single_x))
    J_all_at_theta0_vmap = jit(jax.vmap(J_at_theta0_single_sample, in_axes=(0), out_axes=0))

    ntk3_start_time = time.time()
    J_all_at_theta0 = J_all_at_theta0_vmap(X_train_ntk) # (N, C, P)

    predict_batch_theta0_vmap = jit(jax.vmap(partial(single_sample_forward_flat_params, treedef=treedef_0, shapes_and_dtypes_meta=shapes_meta_0),
                                             in_axes=(None, 0), out_axes=0))
    logits_all_at_theta0 = predict_batch_theta0_vmap(theta_0_flat, X_train_ntk)
    pred_probas_all_at_theta0 = jax_softmax(logits_all_at_theta0)
    Error_all_at_theta0 = pred_probas_all_at_theta0 - Y_train_onehot_ntk

    K0 = compute_empirical_ntk_k0(J_all_at_theta0, num_classes)
    print("NTK3: J, K0, Error0 computed.")

    matrix_B_in_exp = -(2/N_train_ntk) * K0 * (T_factor_ntk3 - 3)
    exp_B_matrix = compute_matrix_exp(matrix_B_in_exp, taylor_order_ntk3)
    
    Identity_N = jnp.eye(N_train_ntk, dtype=K0.dtype)
    Factor_Matrix = Identity_N - exp_B_matrix

    print(f"NTK3: Solving for K0_dagger_times_Factor_times_Error (lambda_reg={lambda_ntk3_reg})...")
    Target_for_solve = jnp.dot(Factor_Matrix, Error_all_at_theta0)
    try:
        K0_dagger_Factor_Error = jnp.linalg.solve(K0 + lambda_ntk3_reg * jnp.eye(N_train_ntk), Target_for_solve)
    except Exception as e:
        print(f"NTK3: jnp.linalg.solve failed: {e}. Attempting with pseudo-inverse.")
        K0_reg_pinv = K0 + lambda_ntk3_reg * jnp.eye(N_train_ntk)
        K0_dagger_Factor_Error = jnp.dot(jnp.linalg.pinv(K0_reg_pinv, rcond=1e-6), Target_for_solve)
    print("NTK3: Solved.")

    param_update_direction = jnp.einsum('ncp,nc->p', J_all_at_theta0, K0_dagger_Factor_Error)
    theta_T_flat = theta_0_flat - param_update_direction
    
    final_params_ntk3 = unflatten_params(theta_T_flat, treedef_0, shapes_meta_0)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    train_loss = float(jax_compute_crossentropy_loss(final_params_ntk3, X_train_ntk, Y_train_onehot_ntk))
    train_acc = float(compute_accuracy_jax(final_params_ntk3, X_train_ntk, Y_train_onehot_ntk))
    val_loss = float(jax_compute_crossentropy_loss(final_params_ntk3, X_val_full, Y_val_onehot_full))
    val_acc = float(compute_accuracy_jax(final_params_ntk3, X_val_full, Y_val_onehot_full))

    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

    ntk3_total_time = time.time() - ntk3_start_time
    print(f"NTK 3 (One-Shot) - Train (full) L: {train_loss:.4f}, A: {train_acc*100:.2f}% | Val (full) L: {val_loss:.4f}, A: {val_acc*100:.2f}%")
    print(f"NTK 3 phase took {ntk3_total_time:.2f} seconds.")
    return final_params_ntk3, history, ntk3_total_time

def evaluate_on_test_jax(params, X_test, Y_test_onehot):
    if params is None:
        print("\nTest Evaluation - No parameters provided.")
        return np.nan, np.nan
    
    test_loss = float(jax_compute_crossentropy_loss(params, X_test, Y_test_onehot))
    test_acc = float(compute_accuracy_jax(params, X_test, Y_test_onehot))

    print(f"\nTest Set Evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

def run_multiseed_scouting(num_scout_runs, master_key):
    all_scout_histories = []
    scout_run_keys = jax.random.split(master_key, num_scout_runs)

    print(f"--- Starting {num_scout_runs}-Seed Scouting Run ---")

    for i in range(num_scout_runs):
        print(f"  Scouting Run {i+1}/{num_scout_runs}...")
        run_key_init, run_key_loop, run_key_ntk = jax.random.split(scout_run_keys[i], 3)

        scout_initial_params = init_network_params(layer_dims, run_key_init)

        X_scout_ntk_subset = X_train_full
        if max_ntk_scouting_val is not None and max_ntk_scouting_val < X_train_full.shape[0]:
            scout_ntk_indices = jax.random.choice(run_key_ntk, X_train_full.shape[0], shape=(max_ntk_scouting_val,), replace=False)
            X_scout_ntk_subset = X_train_full[scout_ntk_indices]

        scout_history = run_sgd_scouting(
            scout_initial_params, X_train_full, Y_train_onehot_full, X_val_full, Y_val_onehot_full,
            scouting_epochs=scouting_epochs,
            batch_size=min(batch_size_config, X_train_full.shape[0]),
            lr_sgd=scouting_lr,
            key_sgd_loop=run_key_loop,
            switch_method=scouting_method,
            X_ntk_monitor_subset=X_scout_ntk_subset,
            num_classes=num_classes,
            param_norm_window=switch_config['param_norm_window'],
            ntk_norm_window=switch_config['ntk_norm_window']
        )
        all_scout_histories.append(scout_history)

    print("\nAggregating scouting results...")
    avg_scout_history = aggregate_histories(all_scout_histories, metric_keys=['val_loss', 'val_acc', 'norm_diff'])

    return avg_scout_history, all_scout_histories[0]
