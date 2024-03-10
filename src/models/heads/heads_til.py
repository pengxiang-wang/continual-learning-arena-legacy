from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn


class HeadsTIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.heads: nn.ModuleList = nn.ModuleList()  # init empty

    def new_task(self, classes: List[Any]) -> None:
        self.output_dim = len(classes)
        self.heads.append(nn.Linear(self.input_dim, self.output_dim))

    def forward(self, feature: torch.Tensor, task_id: int):
        
        if isinstance(task_id, int):
            head = self.heads[task_id]
            logits = head(feature)
        elif isinstance(task_id, torch.Tensor):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logits = torch.empty(0, self.output_dim).to(device)

            for idx in range(feature.size(0)):
                f = feature[idx]
                t = task_id[idx]
                head = self.heads[t]
                logit = head(f).reshape(1,-1)
                logits = torch.cat((logits, logit))
        else:
            logits = None
            
        return logits


if __name__ == "__main__":
    _ = HeadsTIL(input_dim=64)
