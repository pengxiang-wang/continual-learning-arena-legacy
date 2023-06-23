from typing import Any, Dict
import torch
from torch.optim import Optimizer

torch.optim.SGD


class ClipMask(torch.optim.Optimizer):
    def __init__(self, params: _params_t, defaults: Dict[str, Any]) -> None:
        super().__init__(params, defaults)
