from typing import Any, Dict, List, Optional

import torch
from torch import nn


class HeadTIL(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
    ):
        super().__init__()

        self.input_size = input_size
        self.heads: nn.ModuleList = nn.ModuleList()  # init empty

    def new_task(self, classes: List[Any]) -> None:
        output_size = len(classes)
        self.heads.append(nn.Linear(self.input_size, output_size))

    def forward(self, feature: torch.Tensor, task_id: int):
        head = self.heads[task_id]
        logit = head(feature)
        return logit


if __name__ == "__main__":
    _ = HeadTIL(input_size=64)
