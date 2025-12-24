Accelerating Neural Network Training Using the Neural Tangent Kernel
Overview
This repository implements the research code for the Bachelor's thesis "Accelerating Neural Network Training Using the Neural Tangent Kernel" (July 2025). The project investigates a hybrid training strategy that accelerates convergence by transitioning from Stochastic Gradient Descent (SGD) to linearized NTK updates once the network enters the "lazy training" regime.

Key Features
Hybrid Training Engine: Automatically switches from SGD to NTK flow updates based on real-time stability heuristics.

Three NTK Solvers: Implements Discrete Linearized Descent (NTK 1), Functional Evolution (NTK 2), and Integrated Exact Solution (NTK 3).

Generalized Architecture: Supports arbitrary Classification and Regression datasets via configuration files.

JAX Implementation: Utilizes JAX's functional paradigm for efficient Jacobian computation and matrix exponentiation.

Project Structure
The repository is organized as follows: ├── configs/ # YAML configuration files for experiments │ ├── fashion_mnist.yaml # Settings for Classification task │ └── wine_quality.yaml # Settings for Regression task ├── notebooks/ # Jupyter notebooks for demos and figure generation ├── src/ # Source code │ ├── data/ # Data loading and preprocessing logic │ ├── models/ # JAX MLP architecture definition │ ├── ntk/ # Core NTK math (Jacobians, Updates Eq 15-17) │ ├── training/ # Hybrid training loop and stability heuristics │ └── utils/ # Parameter flattening and logging utilities ├── run_experiment.py # Main entry point CLI └── requirements.txt # Python dependencies


## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
Install dependencies:

Bash

pip install -r requirements.txt
Usage
Reproducing Thesis Results
To reproduce the regression results on the Wine Quality dataset:

Bash

python run_experiment.py --config configs/wine_quality.yaml
To reproduce the classification results on Fashion-MNIST:

Bash

python run_experiment.py --config configs/fashion_mnist.yaml
Using Custom Datasets
To use this repository with your own dataset:

Create a new config file in configs/custom_dataset.yaml (see template.yaml).

Implement a loader in src/data/loaders.py.

Run the experiment script pointing to your new config.

Citation
Sarvanau, K. (2025). Accelerating Neural Network Training Using the Neural Tangent Kernel. BSc Thesis, University of Twente.