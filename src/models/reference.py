from typing import Any, Optional

import torch
from torch import nn
from src.models import Finetuning

from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()


class Reference(Finetuning):
    """LightningModule for naive finetuning continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__(heads, backbone, optimizer, scheduler)

    def on_train_start(self):
        for layer in self.backbone.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for head in self.heads.heads:
            for layer in head.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()


if __name__ == "__main__":
    _ = Finetuning(None, None, None, None)
