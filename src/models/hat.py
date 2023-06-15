from typing import Any, Dict, Optional


from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn
from torchmetrics import MeanMetric

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.memories import MaskMemory


class HAT(LightningModule):
    """LightningModule for HAT (Hard Attention to Task) continual learning algorithm."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        s_max: float,
        reg: torch.nn.Module,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.task_id = 0

        self.backbone = backbone
        self.head = head

        self.mask_memory = MaskMemory(self.hparams.s_max, backbone)

        # loss function
        self.reg = reg
        self.criterion = nn.CrossEntropyLoss()

        # special metrics
        self.train_loss_reg = MeanMetric()
        self.val_loss_reg = MeanMetric()
        self.train_loss_total = MeanMetric()
        self.val_loss_total = MeanMetric()

        # manual optimization
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, task_id: int, scalar: float, stage: str):
        # the forward process propagates input to logits of classes of task_id
        feature, mask = self.backbone(x, scalar, stage)
        logits = self.head(feature, task_id)
        return logits, mask

    def on_train_start(self):
        for embedding in self.backbone.te.values():
            # nn.init.constant_(embedding.weight, 1)
            nn.init.uniform_(embedding.weight, -1, 1)

        for logger in self.loggers:
            if type(logger) == TensorBoardLogger:
                tensorboard = logger.experiment
                # tensorboard.add_graph(self.backbone)
                # tensorboard.add_graph(self.head)

    def on_train_end(self):
        self.mask_memory.update(task_id=self.task_id, backbone=self.backbone)

    def annealed_scalar(self, s_max, batch_idx, num_batches):
        s = 1 / s_max + (s_max - 1 / s_max) * (batch_idx - 1) / (num_batches - 1)
        return s

    def clip_mask_gradients(self, previous_mask):
        for module_name, module in self.backbone.named_modules():
            module_name = module_name.replace(".", "")
            if module_name in previous_mask.keys():
                module.weight.grad.data *= 1 - previous_mask[module_name].transpose(
                    0, 1
                )

    def compensate_te_gradients(self, compensate_thres, scalar):
        # Compensate embedding gradients
        for embedding in self.backbone.te.values():
            num = (
                torch.cosh(
                    torch.clamp(
                        scalar * embedding.weight.data,
                        -compensate_thres,
                        compensate_thres,
                    )
                )
                + 1
            )
            den = torch.cosh(embedding.weight.data) + 1
            embedding.weight.grad.data *= self.hparams.s_max / scalar * num / den

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        opt = self.optimizers()
        opt.zero_grad()

        num_batches = self.trainer.num_training_batches
        s = self.annealed_scalar(self.hparams.s_max, batch_idx, num_batches)

        # forward
        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls = self.criterion(logits, y)

        previous_mask = self.mask_memory.get_cumulative_mask()
        loss_reg, _ = self.reg(mask, previous_mask)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # backward
        self.manual_backward(loss_total)
        self.clip_mask_gradients(previous_mask)
        self.compensate_te_gradients(compensate_thres=50, scalar=s)

        opt.step()

        # update and log metrics
        self.train_loss_cls(loss_cls)
        self.train_loss_reg(loss_reg)
        self.train_loss_total(loss_total)
        self.train_acc(preds, targets)
        self.log(
            f"train/loss/cls",
            self.train_loss_cls,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"train/loss/reg",
            self.train_loss_reg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"train/loss/total",
            self.train_loss_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        if batch_idx + 1 == num_batches:
            for logger in self.loggers:
                if type(logger) == TensorBoardLogger:
                    tensorboard = logger.experiment
                    # show mask
                    for module_name, m in mask.items():
                        fig = plt.figure()
                        plt.imshow(m.detach(), aspect=10, cmap="Greys")
                        plt.colorbar()
                        tensorboard.add_figure(f"train/mask/{module_name}", fig)

        # return loss or backpropagation will fail
        return loss_total

    def on_val_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        s = self.hparams.s_max

        logits, mask = self.forward(x, self.task_id, scalar=s, stage="fit")
        loss_cls = self.criterion(logits, y)

        previous_mask = self.mask_memory.get_cumulative_mask()
        loss_reg, _ = self.reg(mask, previous_mask)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        targets = y

        # update and log metrics
        self.val_loss_cls(loss_cls)
        self.val_loss_reg(loss_reg)
        self.val_loss_total(loss_total)
        self.val_acc(preds, targets)
        self.log(
            f"val/loss/cls",
            self.val_loss_cls,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"val/loss/reg",
            self.val_loss_reg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"val/loss/total",
            self.val_loss_total,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            f"val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def on_test_start(self):
        mask = self.mask_memory.get_mask(self.task_id)
        previous_mask = self.mask_memory.get_cumulative_mask()
        for logger in self.loggers:
            if type(logger) == TensorBoardLogger:
                tensorboard = logger.experiment
                # show mask
                for module_name, m in mask.items():
                    fig = plt.figure()
                    plt.imshow(m.detach(), aspect=10, cmap="Greys")
                    plt.colorbar()
                    tensorboard.add_figure(
                        f"test/mask/task{self.task_id}/{module_name}", fig
                    )
                for module_name, m in previous_mask.items():
                    fig = plt.figure()
                    plt.imshow(m.detach(), aspect=10, cmap="Greys")
                    plt.colorbar()
                    tensorboard.add_figure(f"test/mask/previous/{module_name}", fig)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        s = self.hparams.s_max

        test_mask = self.mask_memory.get_mask(dataloader_idx)
        # print(test_mask["fc1"])
        self.backbone.set_test_mask(test_mask)

        logits, _ = self.forward(x, dataloader_idx, scalar=s, stage="test")
        # print(logits)
        loss_cls = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = y

        # update and log metrics
        self.test_loss_cls[dataloader_idx](loss_cls)
        self.test_acc[dataloader_idx](preds, targets)
        self.log(
            f"test/loss/cls",
            self.test_loss_cls[dataloader_idx],
            add_dataloader_idx=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"test/acc",
            self.test_acc[dataloader_idx],
            add_dataloader_idx=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        # test all seen task till self.task_id
        for t in range(self.task_id + 1):
            self.ave_test_loss_cls(self.test_loss_cls[t].compute())
            self.ave_test_acc(self.test_acc[t].compute())
        self.log(f"test/loss/cls/ave", self.ave_test_loss_cls)
        self.log(f"test/acc/ave", self.ave_test_acc)

    def configure_optimizers(self):
        # Choose what optimizers and learning-rate schedulers to use in your optimization
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = HAT(None, None, None, None, None, None)
