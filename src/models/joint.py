from typing import Any, Optional

import torch
from torch import nn

from src.models import Finetuning
from src.utils import logger, logger

log = logger.get_pylogger(__name__)
logger = logger.get_global_logger()


class Joint(Finetuning):
    """LightningModule for joint training continual learning."""

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

    def _model_step_joint(self, batch: Any):
        # common forward step among training, validation, testing step
        x, y, t = batch
        logits = self.forward(x, t)
        loss_cls = self.criterion(logits, y)
        loss_reg = 0.0
        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)

        return loss_cls, loss_reg, loss_total, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_reg, loss_total, preds, targets = self._model_step_joint(batch)

        self.training_step_follow_up(loss_cls, loss_reg, loss_total, preds, targets)

        # return loss or backpropagation will fail
        return loss_total

    def validation_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_reg, loss_total, preds, targets = self._model_step_joint(batch)

        self.validation_step_follow_up(loss_cls, loss_reg, loss_total, preds, targets)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss_cls, _, _, preds, targets = self._model_step(batch, dataloader_idx)

        self.test_step_follow_up(loss_cls, preds, targets, dataloader_idx, batch)


if __name__ == "__main__":
    _ = Finetuning(None, None, None, None)
