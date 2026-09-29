"""Per-buffer transforms applied before whitening.

Network snapshots reference these functions by module path
(buffer_transform.<name>), so keep the module and function names stable.
"""

import numpy as np
import torch

def signed_log1p_transform(x):
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    else:
        return np.sign(x) * np.log1p(np.abs(x))

def depth_transform(x):
    if isinstance(x, torch.Tensor):
        return torch.clamp(torch.log1p(x), min=-1, max=10000)
    else:
        return np.clip(np.log1p(x), -1, 10000)
