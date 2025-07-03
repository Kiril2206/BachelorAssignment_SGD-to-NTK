# BachelorAssignment_SGD-to-NTK

This repository contains the code and analysis for a project exploring the acceleration of neural network training using the Neural Tangent Kernel. The progra is tested on Fashion-MNIST and Wine Quality datasets, however, with correct implementation, it can solve other classification and regression problems. It includes two main notebooks demonstrating each task, along with modularized Python scripts for data loading, model building, and visualization."

# 1. Project structure
- `notebooks/`: Contains the main Jupyter notebooks for analysis.
  - `01-classification-task.ipynb`: Performs the NTK experiments on classification task.
  - `02-regression-task.ipynb`: Performs the NTK experiments on regression task.
- `src/`: Contains modularized Python source code.
  - `shared_utils.py`: Common functions used by both notebooks.
  - `models.py`: Neural network architecture definitions.
  - `data_loader.py`: Functions for fetching and preprocessing data.
  - `plotting.py`: Functions for creating visualizations.
- `results/`: Contains saved outputs like figures and tables.

First, clone the repository and set up the Python environment:

\`\`\`bash
# Clone this repository to your local machine
git clone [https://github.com/your-username/your-project-name.git](https://github.com/Kiril2206/BachelorAssignment_SGD-to-NTK/tree/main)

# Navigate into the cloned project directory
cd BachelorAssignment_SGD-to-NTK

# Create and activate a conda environment
conda env create -f environment.yml
conda activate hybrid_ntk

# Install all the required packages from the requirements file
pip install -r requirements.txt
\`\`\`

# 3. Running the Notebooks

Once the setup is complete, you can launch Jupyter to run the analysis notebooks:

\`\`\`bash
jupyter notebook notebooks/
\`\`\`
This will open a new tab in your web browser. From there, you can open and run `01-classification-task.ipynb` and `02-regression-task.ipynb`.
