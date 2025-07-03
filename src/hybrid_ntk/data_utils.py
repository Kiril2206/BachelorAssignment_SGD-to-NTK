def fetch_split_data(data_name):
    if data_name == ('Fashion_MNIST'):
        print("Fetching Fashion-MNIST data...")
        data = fetch_openml(data_name, version=1, as_frame=False, parser='auto', cache=True)
        
        X_np_all = data.data.astype(np.float32) / 255.0
        Y_labels_np_all = data.target.astype(np.int32)
        num_classes = 10
    
    print("Data fetched and normalized.")

    X_train_val_np, X_test_np, Y_labels_train_val_np, Y_labels_test_np = train_test_split(
            X_np_all, Y_labels_np_all, test_size=0.2, random_state=42, stratify=Y_labels_np_all)
    X_train_np_full, X_val_np_full, Y_labels_train_np_full, Y_labels_val_np_full = train_test_split(
        X_train_val_np, Y_labels_train_val_np, test_size=0.25, random_state=42, stratify=Y_labels_train_val_np)
    
    # Full datasets for SGD and final validation/testing
    X_train_full = jnp.array(X_train_np_full)
    Y_labels_train_full = Y_labels_train_np_full
    X_val_full = jnp.array(X_val_np_full)
    Y_labels_val_full = Y_labels_val_np_full
    X_test_full = jnp.array(X_test_np)
    Y_labels_test_full = Y_labels_test_np
    
    Y_train_onehot_full = jnp.array(one_hot(Y_labels_train_full, num_classes))
    Y_val_onehot_full = jnp.array(one_hot(Y_labels_val_full, num_classes))
    Y_test_onehot_full = jnp.array(one_hot(Y_labels_test_full, num_classes))
    
    print(f"Data shapes: X_train: {X_train_full.shape}, Y_train: {Y_train_onehot_full.shape}")

    return X_train_full, X_val_full, X_test_full, Y_train_onehot_full, Y_val_onehot_full, Y_test_onehot_full

def randomize(num_seeds):
    current_time_seed = 1750980748544

    scouting_key = jax.random.PRNGKey(current_time_seed) 
    scout_key_init, scout_key_loop, scout_key_ntk_subset = jax.random.split(scouting_key, 3)
    
    key_master = jax.random.PRNGKey(current_time_seed + 1)
    keys_for_runs = jax.random.split(key_master, num_seeds)
    
    run_keys = [jax.random.split(k, 7) for k in keys_for_runs]
    
    print(f"Master seed for this run (derived from time): {current_time_seed}")

    return scout_key_loop, run_keys
