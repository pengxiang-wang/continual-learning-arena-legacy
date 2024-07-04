from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models import HATNonUnion
from src.models.calibrators import maskclipper
from src.models.memories import MaskMemory
from src.utils import logger, logger

log = logger.get_pylogger(__name__)
logger = logger.get_global_logger()


DEFAULT_SMAX = 400.0


class HATCompareLoss(HATNonUnion):
    """LightningModule for HAT (Hard Attention to Task) continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        reg: torch.nn.Module,
        s_max: float = DEFAULT_SMAX,
        log_train_mask=False,
    ):
        super().__init__(
            heads, backbone, optimizer, scheduler, reg, s_max, log_train_mask
        )

    def forward(
        self,
        x: torch.Tensor,
        task_id: int,
        scalar: float,
        stage: str,
        additional_mask=None,
    ):
        # the forward process propagates input to logits of classes of task_id
        feature, mask = self.backbone(x, scalar, stage, additional_mask)
        logits = self.heads(feature, task_id)
        return logits, mask

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        num_batches = self.trainer.num_training_batches
        s = self.annealed_scalar(self.hparams.s_max, batch_idx, num_batches)

        # forward step training

        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls_full_forward = self.criterion(logits, y)
        # print("loss_cls_full_forward", loss_cls_full_forward)
        loss_cls = loss_cls_full_forward
        if self.task_id >= 1:
            for t in range(self.task_id):
                logits, mask = self.forward(
                    x,
                    self.task_id,
                    scalar=s,
                    stage="fit",
                    additional_mask=self.mask_memory.get_mask(t),
                )
                loss_cls_part_forward = self.criterion(logits, y)
                # print("loss_cls_part_forward", loss_cls_part_forward)
                loss_cls += (
                    loss_cls_full_forward
                    if loss_cls_full_forward < loss_cls_part_forward
                    else loss_cls_part_forward
                )
            loss_cls = loss_cls / (self.task_id + 1)

            # print("loss_cls", loss_cls)

        previous_masks = self.mask_memory.get_masks()
        loss_reg, _ = self.reg(mask, previous_masks)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # backward step
        self.manual_backward(loss_total)
        previous_mask = self.mask_memory.get_union_mask()
        maskclipper.hard_clip_te_masked_gradients(self.backbone, previous_mask)
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
        logger.log_train_metrics(self, self.train_metrics)

        # log mask
        if self.log_train_mask:
            logger.log_train_mask(
                mask, self.task_id, self.global_step, plot_figure=True
            )

        # return loss or backpropagation will fail
        return loss_total

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        s = self.hparams.s_max

        # forward step validation
        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls = self.criterion(logits, y)

        previous_masks = self.mask_memory.get_masks()
        loss_reg, _ = self.reg(mask, previous_masks)

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


if __name__ == "__main__":
    _ = HAT(None, None, None, None, None, None)
