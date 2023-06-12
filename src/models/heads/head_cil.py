from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from torch import nn


class HeadCIL(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = 0
        self.heads: nn.ModuleList = nn.ModuleList()  # init empty

    def new_task(self, classes: List[Any]):
        output_size_new = len(classes) - self.output_size
        self.heads.append(nn.Linear(self.input_size, output_size_new))
        self.output_size = len(classes)

    def forward(self, feature, task_id=None):
        logits = torch.cat([head(feature) for head in self.heads.values()])
        return logits


if __name__ == "__main__":
    _ = HeadCIL(input_size=64)
