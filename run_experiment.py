import os
import json
import pickle
import argparse
import yaml
import jax
from src.data.factory import get_data_loader
from src.models.mlp import init_network_params, forward
from src.training.engine import run_hybrid_training

def main():
    parser = argparse.ArgumentParser(description="Run Hybrid SGD-NTK Training")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Starting experiment: {config['project_name']}")
    
    # 2. Load Data (Generalized)
    data = get_data_loader(config)
    
    # 3. Initialize Model
    # Construct full layer list: [Input, Hidden..., Output]
    layer_dims = [config['input_dim']] + config['hidden_layers'] + [config['output_dim']]
    key = jax.random.PRNGKey(config['seed'])
    params = init_network_params(layer_dims, key)
    
    # 4. Run Hybrid Training
    # This function (in src.training.engine) manages the SGD -> NTK switch
    final_params, metrics = run_hybrid_training(params, data, config)

    # Create results directory if it doesn't exist
    results_dir = os.path.join("results", "logs")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate a unique filename based on config
    filename = f"{config['project_name']}_{config['ntk_method']}"
    
    # 1. Save Metrics (Loss History) as JSON for easy plotting
    metrics_path = os.path.join(results_dir, f"{filename}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to: {metrics_path}")

    # 2. Save Model Parameters (using Pickle for JAX arrays)
    params_path = os.path.join(results_dir, f"{filename}_params.pkl")
    with open(params_path, 'wb') as f:
        pickle.dump(final_params, f)
    print(f"Params saved to: {params_path}")
    
    print("Training Complete.")

if __name__ == "__main__":
    main()