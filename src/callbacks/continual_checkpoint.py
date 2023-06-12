from typing import Any, Dict

from lightning import Callback, LightningModule, Trainer


class ContinualCheckpoint(Callback):
    """Add task_id information to checkpoints"""

    def __init__(self) -> None:
        super().__init__()

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint["task_id"] = pl_module.task_id

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        pl_module.task_id = checkpoint["task_id"]
