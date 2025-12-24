from.loaders import load_wine_quality, load_fashion_mnist, load_custom

def get_data_loader(config):
    """
    Factory function to dispatch data loading.
    """
    name = config['dataset_name']
    
    if name == 'wine_quality':
        return load_wine_quality(config)
    elif name == 'fashion_mnist':
        return load_fashion_mnist(config)
    elif name == 'custom':
        return load_custom(config)
    else:
        raise ValueError(f"Dataset {name} not supported.")