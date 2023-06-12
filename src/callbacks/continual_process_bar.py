from typing import Any, Dict

from lightning import Callback, LightningModule, Trainer

# I tried to callback or just override the LightningModule method, but not working.

# def get_progress_bar_dict(self):
#     items = super().get_proress_bar_dict()
#     print("XXXXXXXXXXXXXXXXXXXXXx")
#     print(items)
#     items.pop("v_num", None)
#     items['task_id'] = f"{self.task_id:.3d}"
#     return items


def get_progress_bar_dict(pl_module: LightningModule):
    items = pl_module.get_proress_bar_dict()
    items["task_id"] = f"{pl_module.task_id:.3d}"
    return items


class ContinualProgressBar(Callback):
    """Add task_id information to progress bar."""

    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.get_progress_bar_dict = get_progress_bar_dict

    def on_val_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.get_progress_bar_dict = get_progress_bar_dict
