from typing import Any, Dict
import torch
from torch.optim.optimizer import _params_t


class AdaMask(torch.optim.Optimizer):
    def __init__(self, params: _params_t, defaults: Dict[str, Any]) -> None:
        super().__init__(params, defaults)
