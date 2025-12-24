import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_scaler():
    return StandardScaler()

def one_hot_encode(y, num_classes):
    """Encodes labels into one-hot vectors."""
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_reshaped = np.array(y).reshape(-1, 1)
    return encoder.fit_transform(y_reshaped)