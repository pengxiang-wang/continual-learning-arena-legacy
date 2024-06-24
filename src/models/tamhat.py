from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator=".src-root-indicator", pythonpath=True)

from models import HAT
from models.calibrators import maskclipper
from models.memories import MaskMemory
from models.regs import TrapsNMineReg
from utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()


DEFAULT_SMAX = 400.0


class TAMHAT(HAT):
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
        N: int = 100, 
        factor: float = 1.0, 
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
            s_max=self.hparams.s_max, backbone=backbone, approach="tamhat"
        )
                

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        if self.trainer.global_step % self.hparams.N == 0: 
            mask_counter = self.mask_memory.get_mask_counter()
            self.tamreg = TrapsNMineReg(mask_counter, self.hparams.N, self.hparams.factor)
            

        num_batches = self.trainer.num_training_batches
        s = self.annealed_scalar(self.hparams.s_max, batch_idx, num_batches)

        # forward step training
        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls = self.criterion(logits, y)

        previous_mask = self.mask_memory.get_union_mask()
        loss_sparsity_reg, _, reg = self.reg(mask, previous_mask)

        loss_tamreg = self.tamreg(mask)

        loss_total = loss_cls + loss_sparsity_reg + loss_tamreg
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
            loggerpack.log_train_mask(
                mask, self.task_id, self.global_step, plot_figure=True
            )

        # log capacity
        if self.log_capacity:
            loggerpack.log_capacity(capacity, self.task_id, self.global_step)

        self.training_step_follow_up(loss_cls, loss_sparsity_reg, loss_total, preds, targets)

        # return loss or backpropagation will fail
        return loss_total


if __name__ == "__main__":
    _ = TAMHAT(None, None, None, None, None, None)

