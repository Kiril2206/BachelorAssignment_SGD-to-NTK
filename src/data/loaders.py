"""
Specific implementations of data loading.
"""
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml
import numpy as np

def load_wine_quality(config):
    print("Loading Wine Quality Dataset...")
    # ID 186 is Wine Quality in UCI Repo
    wine_quality = fetch_ucirepo(id=186)
    
    # Features and Targets
    X = wine_quality.data.features
    y = wine_quality.data.targets
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_vals = y.values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_vals, test_size=0.2, random_state=config['seed']
    )
    
    return {
        'X_train': jnp.array(X_train),
        'y_train': jnp.array(y_train),
        'X_test': jnp.array(X_test),
        'y_test': jnp.array(y_test)
    }

def load_fashion_mnist(config):
    print("Loading Fashion MNIST...")
    mnist = fetch_openml('Fashion-MNIST', version=1, cache=True, as_frame=False)
    X, y = mnist.data, mnist.target
    
    # Normalize images (0-255 -> 0-1)
    X = X / 255.0
    y = y.astype(int)
    
    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=config['seed']
    )
    
    return {
        'X_train': jnp.array(X_train),
        'y_train': jnp.array(y_train),
        'X_test': jnp.array(X_test),
        'y_test': jnp.array(y_test)
    }

def load_custom(config):
    """
    Placeholder for custom dataset loading.
    Users should implement their own logic here.
    """
    raise NotImplementedError(
        "Custom dataset loading is not yet implemented. "
        "Please define your logic in src/data/loaders.py"
    )