from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import Finetuning
from src.models.calibrators import weightmaskclipper
from src.models.memories import WeightMaskMemory
from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()


class PackNet(Finetuning):
    """LightningModule for PackNet continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        prune_perc: float,
    ):
        super().__init__(heads, backbone, optimizer, scheduler)

        self.prune_perc = prune_perc

        # Memory store mask of each task
        self.weight_mask_memory = WeightMaskMemory(backbone=backbone)

        # manual optimization
        self.automatic_optimization = False

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        # forward step training
        loss_cls, loss_reg, loss_total, preds, targets = self._model_step(
            batch, task_id=self.task_id
        )

        # backward step
        self.manual_backward(loss_total)
        weightmaskclipper.hard_clip_weight_masked_gradients(
            self.backbone, self.weight_mask_memory.get_union_mask()
        )

        opt.step()

        # update metrics
        self.train_metrics[f"task{self.task_id}/train/loss/cls"](loss_cls)
        self.train_metrics[f"task{self.task_id}/train/loss/reg"](loss_reg)
        self.train_metrics[f"task{self.task_id}/train/loss/total"](loss_total)
        self.train_metrics[f"task{self.task_id}/train/acc"](preds, targets)

        # log metrics
        loggerpack.log_train_metrics(self, self.train_metrics)

        # return loss or backpropagation will fail
        return loss_total

    def on_train_end(self):

        # select and prun
        current_mask = weightmaskclipper.prune(
            self.backbone,
            self.weight_mask_memory.get_union_mask(),
            self.task_id,
            self.prune_perc,
        )
        self.weight_mask_memory.update(current_mask)

        # retrain

        # for epoch in smaller_epochs:
        #    self.backbone. only forwardsuse


if __name__ == "__main__":
    _ = PackNet(None, None, None, None, None)
