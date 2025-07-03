def one_hot(y, num_classes):
    y_int = np.asarray(y, dtype=int)
    return np.eye(num_classes)[y_int.reshape(-1)]
