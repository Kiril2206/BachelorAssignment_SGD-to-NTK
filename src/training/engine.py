import time
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
from src.models.mlp import forward 
from src.training.heuristics import check_parameter_stability, check_ntk_stability
from src.ntk.dynamics import compute_jacobian, compute_ntk_matrix
from src.ntk.updates import ntk_update_1_linearized, ntk_update_2_functional, ntk_update_3_integrated
from src.utils.flattening import flatten_params, get_shapes_and_dtypes, unflatten_params
from src.training.objectives import get_loss_fn

def run_hybrid_training(params, data, config):
    X_train, y_train = data['X_train'], data['y_train']
    
    learning_rate = config['learning_rate']
    loss_fn_core = get_loss_fn(config['problem_type'])
    
    @jit
    def step_sgd(params, x, y):
        val, grads = value_and_grad(loss_fn_core)(params, x, y, lambda p, x: forward(p, x))
        new_params = [(w - learning_rate * dw, b - learning_rate * db) 
                      for (w, b), (dw, db) in zip(params, grads)]
        return new_params, val

    param_history = []
    loss_history = []
    start_time = time.time()
    switched = False
    ntk_meta = {} 

    print(f"Starting Training ({config['max_epochs']} epochs)...")

    for epoch in range(config['max_epochs']):
        
        # --- PHASE 1: SGD or NTK-Iterative ---
        if not switched:
            params, loss_val = step_sgd(params, X_train, y_train)
            loss_history.append(float(loss_val))
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Loss = {loss_val:.4f}")

            # --- PHASE 2: Check Heuristics ---
            param_history.append(params)
            
            if len(param_history) > config['heuristics']['window']:
                oldest_params = param_history.pop(0)
                is_stable, metric = check_parameter_stability(params, oldest_params, config['heuristics']['threshold'])
                
                if is_stable:
                    print(f"--> STABILITY REACHED at Epoch {epoch} (Metric: {metric:.5f})")
                    print("--> Switching to NTK Mode...")
                    switched = True
                    
                    # --- PHASE 3: Prepare NTK ---
                    sample_limit = config.get('ntk_sample_size', None)
                    if sample_limit is not None and sample_limit < len(X_train):
                        print(f"    Using subset of {sample_limit} samples for Jacobian.")
                        X_ntk = X_train[:sample_limit]
                        y_ntk = y_train[:sample_limit]
                    else:
                        X_ntk = X_train
                        y_ntk = y_train

                    print("    Computing Jacobian (J0) and NTK Matrix (Theta0)...")
                    flat_params, treedef = flatten_params(params)
                    shapes = get_shapes_and_dtypes(params)
                    
                    J0 = compute_jacobian(flat_params, X_ntk, treedef, shapes)
                    Theta0 = compute_ntk_matrix(J0)
                    
                    ntk_meta = {
                        'J0': J0,
                        'Theta0': Theta0,
                        'flat_params_0': flat_params,
                        'treedef': treedef,
                        'shapes': shapes,
                        'subset_y': y_ntk,
                        'subset_X': X_ntk
                    }
                    ntk_meta['f_0'] = forward(params, ntk_meta['subset_X'])

                    # IF NTK 3 (One-Shot)
                    if config['ntk_method'] == 'ntk_3_integrated':
                        print("    Applying One-Shot NTK Solution (Eq 17)...")
                        remaining_epochs = config['max_epochs'] - epoch
                        
                        flat_final = ntk_update_3_integrated(
                            ntk_meta['flat_params_0'], 
                            ntk_meta['J0'], 
                            ntk_meta, # <--- FIX: Corrected from passing the dict
                            ntk_meta['f_0'], 
                            ntk_meta['subset_y'], 
                            remaining_epochs, 
                            len(X_train) 
                        )
                        params = unflatten_params(flat_final, treedef, shapes)
                        print("    One-Shot Update Complete.")
                        break 

        else:
            # --- PHASE 4: Post-Switch Updates (NTK 1 or 2) ---
            flat_curr, _ = flatten_params(params)
            
            if config['ntk_method'] == 'ntk_1_linearized':
                 f_curr = forward(params, ntk_meta['subset_X'])
                 flat_new = ntk_update_1_linearized(
                     flat_curr, 
                     ntk_meta['J0'], 
                     f_curr, 
                     ntk_meta['subset_y'], 
                     learning_rate, 
                     len(X_train)
                 )
                 params = unflatten_params(flat_new, ntk_meta['treedef'], ntk_meta['shapes'])
            
            if epoch % 1 == 0:
                 # Optional: Evaluate on the NTK subset to see if it's working
                 loss_ntk = get_loss_fn(config['problem_type'])(params, ntk_meta['subset_X'], ntk_meta['subset_y'], lambda p, x: forward(p, x))
                 print(f"Epoch {epoch} (NTK): Loss on subset = {loss_ntk:.4f}")

    print(f"Training finished in {time.time() - start_time:.2f}s")
    return params, {'loss': loss_history}