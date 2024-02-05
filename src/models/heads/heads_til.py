from typing import Any, Dict, List, Optional

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
        output_dim = len(classes)
        self.heads.append(nn.Linear(self.input_dim, output_dim))

    def forward(self, feature: torch.Tensor, task_id: int):
        head = self.heads[task_id]
        logit = head(feature)
        return logit


if __name__ == "__main__":
    _ = HeadsTIL(input_dim=64)
