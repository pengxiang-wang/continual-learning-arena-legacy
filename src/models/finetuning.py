from typing import Any, Optional

import torch
from torch import nn
from lightning import LightningModule

# import our own modules
# because of the setup_root in train.py and so on, we can import from src without any problems
from src.utils import get_logger
logger = get_logger()


class Finetuning(LightningModule):
    r"""LightningModule for naive finetuning continual learning algorithm.
    #
    To apply this algorithm, specify `_targets_` in `configs/model` and specify other parameters in the same config file.

    Args:
        heads: output heads for continual learning tasks, determining TIL or CIL scenario. Defined in `src/models/heads`
        backbone: network before heads, shared across tasks.Defined in `src/models/backbones`
        optimizer: optimizer for training each task.
        scheduler: learning rate scheduler for training each task.
    """

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self maintained task_id counter
        self.task_id: Optional[int] = None

        # store network module in 'self' beyond 'self.hparams' for convenience
        self.backbone = backbone
        self.heads = heads

        # loss function
        self.criterion = nn.CrossEntropyLoss()  # classification loss

    def forward(self, x: torch.Tensor, task_id: int):
        # forward propagation for task `task_id`, from inputs to logits
        # nothing related to forward function in PyTorch. Just for convenience.
        feature = self.backbone(x)
        logits = self.heads(feature, task_id)
        return logits

    def _model_step(self, batch: Any, task_id: int):
        # common forward step among training, validation, testing step
        x, y = batch
        logits = self.forward(x, task_id)
        loss_cls = self.criterion(logits, y)
        loss_total = loss_cls
        preds = torch.argmax(logits, dim=1)

        return loss_cls, loss_total, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_total, preds, targets = self._model_step(
            batch, task_id=self.task_id
        )

        self.training_step_follow_up(loss_cls, loss_total, preds, targets)

        # return loss or backpropagation will fail
        return loss_total

    def training_step_follow_up(self, loss_cls, loss_total, preds, targets):

        # update metrics
        self.train_metrics[f"task{self.task_id}/train/loss/cls"](loss_cls)
        self.train_metrics[f"task{self.task_id}/train/loss/total"](loss_total)
        self.train_metrics[f"task{self.task_id}/train/acc"](preds, targets)

        # log_metrics
        logger.log_train_metrics(self, self.train_metrics)

    def on_val_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics[f"task{self.task_id}/val/loss/cls"].reset()
        self.val_metrics[f"task{self.task_id}/val/loss/total"].reset()
        self.val_metrics[f"task{self.task_id}/val/acc"].reset()
        self.val_metrics[f"task{self.task_id}/val/acc/best"].reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_total, preds, targets = self._model_step(
            batch, task_id=self.task_id
        )

        self.validation_step_follow_up(loss_cls, loss_total, preds, targets)

    def validation_step_follow_up(self, loss_cls, loss_total, preds, targets):

        # update metrics
        self.val_metrics[f"task{self.task_id}/val/loss/cls"](loss_cls)
        self.val_metrics[f"task{self.task_id}/val/loss/total"](loss_total)
        self.val_metrics[f"task{self.task_id}/val/acc"](preds, targets)

        # log metrics
        logger.log_val_metrics(self, self.val_metrics)

    def on_validation_epoch_end(self):
        acc = self.val_metrics[
            f"task{self.task_id}/val/acc"
        ].compute()  # get current val acc
        self.val_metrics[f"task{self.task_id}/val/acc/best"](
            acc
        )  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            f"task{self.task_id}/val/acc_best",
            self.val_metrics[f"task{self.task_id}/val/acc/best"].compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss_cls, _, preds, targets = self._model_step(batch, dataloader_idx)

        self.test_step_follow_up(loss_cls, preds, targets, dataloader_idx, batch)

    def test_step_follow_up(self, loss_cls, preds, targets, dataloader_idx, batch):

        # update metrics
        self.test_metrics["test/loss/cls"][dataloader_idx](loss_cls)
        self.test_metrics["test/acc"][dataloader_idx](preds, targets)

        # log metrics
        logger.log_test_metrics_progress_bar(
            self, self.test_metrics, dataloader_idx
        )

        logger.log_test_samples(batch, preds, targets, dataloader_idx)

    def on_test_epoch_end(self):
        # update metrics
        for t in range(self.task_id + 1):
            self.test_metrics_overall[f"test/loss/cls/ave"](
                self.test_loss_cls[t].compute()
            )
            self.test_metrics_overall[f"test/acc/ave"](self.test_acc[t].compute())
            # self.test_metrics_overall[f"test/bwt"](self.test_acc[t].compute())

        # log metrics
        logger.log_test_metrics(
            self, self.test_metrics, self.test_metrics_overall, task_id=self.task_id
        )

    def predict(self, batch: Any, task_id: int):
        """Pure prediction.

        Returns:
        preds (Tensor): predicted classes of batch.
        prods (Tensor): scores of predicted classes of batch.
        """
        logits = self.forward(batch, task_id)
        probs, preds = torch.max(logits, dim=1)
        return preds, probs

    def configure_optimizers(self):
        # choose optimizers
        optimizer = self.hparams.optimizer(params=self.parameters())

        # choose learning-rate schedulers
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": f"task{self.task_id}/val/loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Finetuning(None, None, None, None)
