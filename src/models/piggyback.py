from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models import Finetuning
from src.models.calibrators import weightmaskclipper
from src.models.memories import WeightMaskMemory
from src.utils import logger, logger

log = logger.get_pylogger(__name__)
logger = logger.get_global_logger()


class Piggyback(Finetuning):
    """LightningModule for Piggyback continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        threshold: float,
    ):
        super().__init__(heads, backbone, optimizer, scheduler)

        # Memory store mask of each task
        self.weight_mask_memory = WeightMaskMemory(
            backbone=backbone, datatype="real", threshold=self.hparams.threshold
        )

        # manual optimization
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, task_id: int, stage: str):
        # the forward process propagates input to logits of classes of task_id
        feature = self.backbone(x, stage)
        logits = self.heads(feature, task_id)
        return logits

    def on_train_start(self):
        # for n,p in self.backbone.named_parameters():
        # nn.init.normal_(self.backbone.mask[n], 0, 1)
        pass

    def on_train_end(self):
        self.weight_mask_memory.update(task_id=self.task_id, backbone=self.backbone)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        # forward step training
        logits = self.forward(x, self.task_id, stage="fit")
        loss_cls = self.criterion(logits, y)

        loss_reg = 0.0

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # backward step
        self.manual_backward(loss_total)
        # for n,p in self.backbone.named_parameters():
        # if n == "fc.0.weight":
        # print("p.data grad", p.grad)

        # print("grad", self.backbone.params["fc.0.weight"].grad)

        opt.step()
        # self.backbone.take_off_mask()

        # update metrics
        self.train_metrics[f"task{self.task_id}/train/loss/cls"](loss_cls)
        self.train_metrics[f"task{self.task_id}/train/loss/reg"](loss_reg)
        self.train_metrics[f"task{self.task_id}/train/loss/total"](loss_total)
        self.train_metrics[f"task{self.task_id}/train/acc"](preds, targets)

        # log metrics
        logger.log_train_metrics(self, self.train_metrics)

        # return loss or backpropagation will fail
        return loss_total

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        # forward step validation
        logits = self.forward(x, self.task_id, stage="fit")
        loss_cls = self.criterion(logits, y)

        loss_reg = 0.0

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # update metrics
        self.val_metrics[f"task{self.task_id}/val/loss/cls"](loss_cls)
        self.val_metrics[f"task{self.task_id}/val/loss/reg"](loss_reg)
        self.val_metrics[f"task{self.task_id}/val/loss/total"](loss_total)
        self.val_metrics[f"task{self.task_id}/val/acc"](preds, targets)

        # log metrics
        logger.log_val_metrics(self, self.val_metrics)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        test_mask = self.weight_mask_memory.get_mask(dataloader_idx)
        self.backbone.set_test_mask(test_mask)

        logits = self.forward(x, dataloader_idx, stage="test")
        loss_cls = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = y

        # update metrics
        self.test_metrics["test/loss/cls"][dataloader_idx](loss_cls)
        self.test_metrics["test/acc"][dataloader_idx](preds, targets)

        # log metrics
        logger.log_test_metrics_progress_bar(self, self.test_metrics, dataloader_idx)

        logger.log_test_samples(batch, preds, targets, dataloader_idx)


if __name__ == "__main__":
    _ = Piggyback(None, None, None, None, None)
