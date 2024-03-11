import os
from typing import Any, Dict, List, Tuple

from lightning import LightningModule, LightningDataModule, Trainer
import lightning.pytorch as pl
import pyrootutils
import torch
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def task_labeled(Dataset):
    class DatasetTaskLabeled(Dataset):

        def __init__(self, task_id: int, *args, **kw):
            
            super().__init__(*args, **kw)
            self.task_label = task_id
            
            self.__class__.__name__ = Dataset.__name__
            
        def __getitem__(self, index: int):
            
            x, y = super().__getitem__(index)
            return x, y, self.task_label

    return DatasetTaskLabeled



def set_task_train(
    task_id: int,
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
) -> None:
    """Set data and model lightning modules to new task."""

    # maintain task_id counter
    datamodule.task_id = task_id
    model.task_id = task_id
    trainer.task_id = task_id

    test_task_id_list = range(task_id + 1)
    
    # add new head
    classes = datamodule.classes(task_id)
    if hasattr(model, "heads"):
        model.heads.new_task(classes)

    num_classes = len(classes)
    num_classes_total = [len(datamodule.classes(t)) for t in test_task_id_list]

    # setup metrics
    # train metrics (single task)
    exec(f"model.task{task_id}_train_loss_cls = MeanMetric()")
    exec(f"model.task{task_id}_train_loss_reg = MeanMetric()")
    exec(f"model.task{task_id}_train_loss_total = MeanMetric()")
    exec(
        f"model.task{task_id}_train_acc = Accuracy(task='multiclass', num_classes=num_classes)"
    )
    train_metrics = {}  # name dict
    train_metrics[f"task{task_id}/train/loss/cls"] = eval(
        f"model.task{task_id}_train_loss_cls"
    )
    train_metrics[f"task{task_id}/train/loss/reg"] = eval(
        f"model.task{task_id}_train_loss_reg"
    )
    train_metrics[f"task{task_id}/train/loss/total"] = eval(
        f"model.task{task_id}_train_loss_total"
    )
    train_metrics[f"task{task_id}/train/acc"] = eval(f"model.task{task_id}_train_acc")
    model.train_metrics = train_metrics
    # val metrics (single task)
    exec(f"model.task{task_id}_val_loss_cls = MeanMetric()")
    exec(f"model.task{task_id}_val_loss_reg = MeanMetric()")
    exec(f"model.task{task_id}_val_loss_total = MeanMetric()")
    exec(
        f"model.task{task_id}_val_acc = Accuracy(task='multiclass', num_classes=num_classes)"
    )
    exec(
        f"model.task{task_id}_val_acc_best = MaxMetric()"
    )  # for tracking best so far validation accuracy
    val_metrics = {}  # name dict
    val_metrics[f"task{task_id}/val/loss/cls"] = eval(
        f"model.task{task_id}_val_loss_cls"
    )
    val_metrics[f"task{task_id}/val/loss/reg"] = eval(
        f"model.task{task_id}_val_loss_reg"
    )
    val_metrics[f"task{task_id}/val/loss/total"] = eval(
        f"model.task{task_id}_val_loss_total"
    )
    val_metrics[f"task{task_id}/val/acc"] = eval(f"model.task{task_id}_val_acc")
    val_metrics[f"task{task_id}/val/acc/best"] = eval(
        f"model.task{task_id}_val_acc_best"
    )
    model.val_metrics = val_metrics
    # test metrics (single task)
    model.test_loss_cls = nn.ModuleList([MeanMetric() for _ in test_task_id_list])
    model.test_acc = nn.ModuleList(
        [
            Accuracy(task="multiclass", num_classes=num_classes)
            for num_classes in num_classes_total
        ]
    )
    test_metrics = {}
    test_metrics[f"test/loss/cls"] = model.test_loss_cls
    test_metrics[f"test/acc"] = model.test_acc
    model.test_metrics = test_metrics
    # test metrics (overall across task)
    model.test_loss_cls_ave = MeanMetric()
    model.test_acc_ave = MeanMetric()
    # model.test_bwt = None # not implemented, have to inherit from last
    test_metrics_overall = {}
    test_metrics_overall[f"test/loss/cls/ave"] = model.test_loss_cls_ave
    test_metrics_overall[f"test/acc/ave"] = model.test_acc_ave
    # test_metrics_overall[f"test/bwt"] = model.test_bwt
    model.test_metrics_overall = test_metrics_overall

    # setup callbacks
    if trainer.checkpoint_callback:
        trainer.checkpoint_callback.dirpath = os.path.join(
            trainer.checkpoint_callback.dirpath, f"task{task_id}"
        )  # seperate task output directory
        trainer.checkpoint_callback.monitor = (
            f"task{task_id}/{trainer.checkpoint_callback.monitor}"
        )
    if trainer.early_stopping_callback:
        trainer.early_stopping_callback.monitor = (
            f"task{task_id}/{trainer.early_stopping_callback.monitor}"
        )


def set_test(
    datamodule: LightningDataModule,
    model: LightningModule,
    ckpt_path,
):
    """Set data and model lightning modules to test task.
    
    Args:
        task_id_list (List[int]): List of task ids to be tested.
    """

    model.task_id = torch.load(ckpt_path)["task_id"]
    num_tasks_ckpt = model.task_id + 1
    task_id_list = range(num_tasks_ckpt)

    # quick setup of datamodule and heads
    for t in task_id_list:
        datamodule.task_id = t
        datamodule.setup(stage="test")
        classes = datamodule.classes(t)
        model.heads.new_task(classes)

    num_classes_total = [len(datamodule.classes(t)) for t in task_id_list]

    # setup metrics
    # test metrics (across task)    
    model.test_loss_cls = nn.ModuleList([MeanMetric() for _ in task_id_list])
    model.test_acc = nn.ModuleList(
        [
            Accuracy(task="multiclass", num_classes=num_classes)
            for num_classes in num_classes_total
        ]
    )
    test_metrics = {}
    test_metrics[f"test/loss/cls"] = model.test_loss_cls
    test_metrics[f"test/acc"] = model.test_acc
    model.test_metrics = test_metrics
    # test metrics (overall across task)
    model.test_loss_cls_ave = MeanMetric()
    model.test_acc_ave = MeanMetric()
    # model.test_bwt = None # not implemented, have to inherit from last
    test_metrics_overall = {}
    test_metrics_overall[f"test/loss/cls/ave"] = model.test_loss_cls_ave
    test_metrics_overall[f"test/acc/ave"] = model.test_acc_ave
    # test_metrics_overall[f"test/bwt"] = model.test_bwt
    model.test_metrics_overall = test_metrics_overall


def set_predict(
    datamodule: LightningDataModule,
    model: LightningModule,
    ckpt_path,
):
    """Set data and model lightning modules to test task.
    
    Args:
        task_id_list (List[int]): List of task ids to be tested.
    """
    model.eval() # manually set eval mode because predict.py doesn't use Lightning predicting APIs.

    model.task_id = torch.load(ckpt_path)["task_id"]
    num_tasks_ckpt = model.task_id + 1
    task_id_list = range(num_tasks_ckpt)

    # quick setup of datamodule and heads
    for t in task_id_list:
        datamodule.task_id = t
        datamodule.setup(stage="test")
        classes = datamodule.classes(t)
        model.heads.new_task(classes)
        
        
# def distribute_task_train_val_test_split(
#     train_val_test_split: Tuple[int, int, int], num_data: int, num_data_task: int
# ):
#     pc = num_data_task / num_data
#     print()
#     train_val_test_split_task = [int(pc * num) for num in train_val_test_split]
#     train_val_test_split_task[0] = num_data_task - sum(train_val_test_split_task[1:])
#     train_val_test_split_task = tuple(train_val_test_split_task)
#     return train_val_test_split_task
