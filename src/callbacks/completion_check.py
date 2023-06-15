from typing import Any, Dict

from lightning import Callback, LightningModule, Trainer
from omegaconf import DictConfig


# Unfinished. Maybe cleanup.sh doesn't need a preprocessed flag


class CompletionCheck(Callback):
    """Check if the experiment completes. Assign with a flag for cleanup.sh"""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.output_dir = cfg.paths.output_dir
