import torch
import numpy as np

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.clone().detach().float()