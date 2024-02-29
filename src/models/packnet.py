from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import Finetuning
from src.models.memories import ModelMemory
from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()


class PackNet(Finetuning):
    """LightningModule for LwF (Learning without Forgetting) continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        prune_rate: float,
    ):
        super().__init__(heads, backbone, optimizer, scheduler)

        # Memory store mask of each task
        self.weight_mask_memory = WeightMaskMemory(parameters=backbone.parameters())
        

    def on_train_end(self):
        
        # select and prun
        
        # retrain
        

if __name__ == "__main__":
    _ = PackNet(None, None, None, None, None)
