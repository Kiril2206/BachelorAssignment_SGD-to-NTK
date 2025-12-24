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
    
    print("Training Complete.")

if __name__ == "__main__":
    main()