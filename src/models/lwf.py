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


class LwF(Finetuning):
    """LightningModule for LwF (Learning without Forgetting) continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        reg: torch.nn.Module,
    ):
        super().__init__(heads, backbone, optimizer, scheduler)

        # LwF regularisation loss function
        self.reg = reg

        # Memory store mask of each task
        self.model_memory = ModelMemory()
        

    def on_train_end(self):
        self.model_memory.update(task_id=self.task_id, backbone=self.backbone, heads=self.heads)


    def _model_step(self, batch: Any, task_id: int):
        # common forward step among training, validation, testing step
        x, y = batch
        logits = self.forward(x, task_id)
        loss_cls = self.criterion(logits, y)
        
        loss_reg = 0.0
        for task_id in range(self.task_id):
            teachers_old = self.model_memory.forward(x, task_id)
            loss_reg += self.reg(logits, teachers_old) 
        
        print(loss_reg)
        
        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)

        return loss_cls, loss_reg, loss_total, preds, y

if __name__ == "__main__":
    _ = LwF(None, None, None, None, None)
