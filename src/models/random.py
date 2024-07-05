from typing import Any, Optional

import torch
from torch import nn
from lightning import LightningModule


from src.models.memories import LabelMemory
from src.utils import get_logger

logger = get_logger()


class Random(LightningModule):
    """LightningModule for continual learning with random."""

    def __init__(self):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self maintained task_id counter
        self.task_id: Optional[int] = None

        self.train_label_memory = LabelMemory()

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def on_train_batch_end(self, output, batch: Any, batch_idx: int):
        # save
        self.train_label_memory.update(
            batch, self.task_id
        )  # for calculating fisher information

        loss_total = 0.0

        # return loss or backpropagation will fail
        return loss_total

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):

        _, targets = batch
        num = len(targets)
        preds = self.train_label_memory.get_random_labels(dataloader_idx, num)

        if dataloader_idx == 1:
            pass
        # update metrics
        self.test_metrics["test/acc"][dataloader_idx](preds, targets)

        # log metrics
        logger.log_test_metrics_progress_bar(self, self.test_metrics, dataloader_idx)

        logger.log_test_samples(batch, preds, targets, dataloader_idx)

    def on_test_epoch_end(self):
        # update metrics
        for t in range(self.task_id + 1):
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
        _, targets = batch
        num = len(targets)
        preds = self.train_label_memory.get_random_labels(task_id, num)
        probs = None
        return preds, probs

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    _ = Random(None, None, None, None)
