from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from torch import nn


class HeadsCIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 0
        self.heads: nn.ModuleList = nn.ModuleList()  # init empty

    def new_task(self, classes: List[Any]):
        output_dim_new = len(classes) - self.output_dim
        self.heads.append(nn.Linear(self.input_dim, output_dim_new))
        self.output_dim = len(classes)

    def forward(self, feature, task_id=None):
        # always use all the heads regardless of task id. Therefore CIL predicts without the knowledge of task id
        logits = torch.cat([head(feature) for head in self.heads], dim=-1)
        return logits


if __name__ == "__main__":
    _ = HeadsCIL(input_dim=64)
