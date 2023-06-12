import os
from typing import Any, Dict

from lightning import LightningModule, LightningDataModule
import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


def set_task_train(
    task_id: int,
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer,
    cfg,
) -> None:
    """Set data and model lightning modules to new task."""
    task_names = datamodule.task_names
    task_name = task_names[task_id] if task_names else str(task_id)

    datamodule.task_name = task_name

    datamodule.task_id = task_id
    model.task_id = task_id

    classes = datamodule.classes(task_id)
    model.head.new_task(classes)

    num_classes = len(classes)
    num_classes_total = [len(datamodule.classes(t)) for t in range(task_id + 1)]

    # set metrics

    # metric objects for calculating and averaging accuracy across batches and tasks
    model.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
    model.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
    model.test_acc = nn.ModuleList(
        [
            Accuracy(task="multiclass", num_classes=num_classes)
            for num_classes in num_classes_total
        ]
    )
    model.ave_test_acc = MeanMetric()

    # for averaging loss across batches
    model.train_loss = MeanMetric()
    model.val_loss = MeanMetric()
    model.test_loss = nn.ModuleList([MeanMetric() for _ in num_classes_total])
    model.ave_test_loss = MeanMetric()

    # for tracking best so far validation accuracy
    model.val_acc_best = MaxMetric()

    if trainer.checkpoint_callback:
        trainer.checkpoint_callback.dirpath = os.path.join(
            cfg.paths.get("output_dir"), f"task{task_id}"
        )

    if trainer.early_stopping_callback:
        pass


def set_test(
    datamodule: LightningDataModule,
    model: LightningModule,
    ckpt_path,
):
    num_tasks_ckpt = torch.load(ckpt_path)["task_id"] + 1

    # quick setup of datamodule and heads
    for t in range(num_tasks_ckpt):
        datamodule.task_id = t
        datamodule.setup(stage="test")
        classes = datamodule.classes(t)
        model.head.new_task(classes)

    num_classes_total = [len(datamodule.classes(t)) for t in range(num_tasks_ckpt)]

    # metric objects for calculating and averaging accuracy across batches and tasks
    model.test_acc = nn.ModuleList(
        [
            Accuracy(task="multiclass", num_classes=num_classes)
            for num_classes in num_classes_total
        ]
    )
    model.ave_test_acc = MeanMetric()

    # for averaging loss across batches
    model.test_loss = nn.ModuleList([MeanMetric() for _ in num_classes_total])
    model.ave_test_loss = MeanMetric()
