from typing import Any, Dict

from lightning import Callback, LightningModule, Trainer


class CompletionCheck(Callback):
    """Check if the experiment completes. Assign with a flag for cleanup.sh"""

    def __init__(self) -> None:
        super().__init__()
