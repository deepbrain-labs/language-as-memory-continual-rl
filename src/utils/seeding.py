import random
import numpy as np
import torch

def set_global_seeds(seed: int):
    """
    Sets the random seeds for Python's random, NumPy, and PyTorch (CPU & GPU)
    to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seeds set to {seed}")
