from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import HAT
from src.models.calibrators import maskclipper
from src.models.memories import MaskMemory
from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()


DEFAULT_SMAX = 400.0


class AdaHAT(HAT):
    """LightningModule for AdaHAT (Hard Attention to Task) continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        reg: torch.nn.Module,
        s_max: float = DEFAULT_SMAX,
        adjust_rate: float = 1e-06,
        log_train_mask=False,
    ):
        super().__init__(
            heads, backbone, optimizer, scheduler, reg, s_max, log_train_mask
        )

        # Memory store mask of each task
        self.mask_memory = MaskMemory(
            s_max=self.hparams.s_max, backbone=backbone, approach="adahat"
        )

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        num_batches = self.trainer.num_training_batches
        s = self.annealed_scalar(self.hparams.s_max, batch_idx, num_batches)

        # forward step training
        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls = self.criterion(logits, y)

        previous_mask = self.mask_memory.get_union_mask()
        loss_reg, _, reg = self.reg(mask, previous_mask)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # backward step
        self.manual_backward(loss_total)
        previous_mask_sum = self.mask_memory.get_sum_mask()
        maskclipper.soft_clip_te_masked_gradients(
            self.backbonae,
            previous_mask_sum,
            previous_mask,
            reg,
            self.hparams.adjust_rate,
        )
        maskclipper.compensate_te_gradients(
            self.backbone, compensate_thres=50, scalar=s, s_max=self.hparams.s_max
        )
        opt.step()

        # update metrics
        self.train_metrics[f"task{self.task_id}/train/loss/cls"](loss_cls)
        self.train_metrics[f"task{self.task_id}/train/loss/reg"](loss_reg)
        self.train_metrics[f"task{self.task_id}/train/loss/total"](loss_total)
        self.train_metrics[f"task{self.task_id}/train/acc"](preds, targets)

        # log_metrics
        loggerpack.log_train_metrics(self, self.train_metrics)

        # log mask
        if self.log_train_mask:
            loggerpack.log_train_mask(
                mask, self.task_id, self.global_step, plot_figure=True
            )

        # return loss or backpropagation will fail
        return loss_total


if __name__ == "__main__":
    _ = HAT(None, None, None, None, None, None)
