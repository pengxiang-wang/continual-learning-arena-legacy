from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models import HAT
from src.models.calibrators import maskclipper
from src.models.memories import MaskMemory
from src.utils import logger, logger

log = logger.get_pylogger(__name__)
logger = logger.get_global_logger()


DEFAULT_SMAX = 400.0


class AdaHAT(HAT):
    """LightningModule for AdaHAT (Hard Attention to Task) continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        mask_sparsity_reg: torch.nn.Module,
        te_init: str = "N01",
        s_max: float = DEFAULT_SMAX,
        adjust_strategy: str = "ada",
        alpha: float = 1e-06,
        calculate_capacity: bool = False,
        log_capacity: bool = False,
        log_train_mask=False,
    ):
        super().__init__(
            heads,
            backbone,
            optimizer,
            scheduler,
            mask_sparsity_reg,
            s_max,
            calculate_capacity,
            log_capacity,
            log_train_mask,
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
        loss_reg, _, reg = self.mask_sparsity_reg(mask, previous_mask)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # backward step
        self.manual_backward(loss_total)

        capacity = maskclipper.soft_clip_te_masked_gradients(
            self.backbone,
            self.hparams.adjust_strategy,
            self.mask_memory,
            self.task_id,
            reg,
            self.hparams.alpha,
            self.log_capacity,
        )
        maskclipper.compensate_te_gradients(
            self.backbone, compensate_thres=50, scalar=s, s_max=self.hparams.s_max
        )
        # torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)
        opt.step()

        # log mask
        if self.log_train_mask:
            logger.log_train_mask(
                mask, self.task_id, self.global_step, plot_figure=True
            )

        # log capacity
        if self.log_capacity:
            logger.log_capacity(capacity, self.task_id, self.global_step)

        self.training_step_follow_up(loss_cls, loss_reg, loss_total, preds, targets)

        # return loss or backpropagation will fail
        return loss_total


if __name__ == "__main__":
    _ = AdaHAT(None, None, None, None, None, None)
