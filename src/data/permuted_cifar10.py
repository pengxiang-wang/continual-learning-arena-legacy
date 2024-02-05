from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10 as OrigDataset
from torchvision.transforms import transforms

from src.data import transforms as my_transforms
from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()

NUM_CLASSES = 10
INPUT_SIZE = (3, 32, 32)
CHANNEL_SIZE = 32 * 32
MEAN = (0.49139968, 0.48215827 ,0.44653124)
STD = (0.24703233, 0.24348505, 0.26158768)

DEFAULT_NUM_TASKS = 10
DEFAULT_PERM_SEEDS = [s for s in range(DEFAULT_NUM_TASKS)]


class PermutedCIFAR10(LightningDataModule):
    """LightningDataModule for Pemuted MNIST dataset.

    TIL (Task-Incremental Learning) scenario. Must use HeadsTIL for your model.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        scenario: str = "TIL",
        num_tasks: int = DEFAULT_NUM_TASKS,
        perm_seeds: List[int] = DEFAULT_PERM_SEEDS,
        val_pc: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # meta info
        self.data_dir = data_dir
        self.scenario = scenario

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test_orig: Dict[int, Optional[Dataset]] = {}
        self.data_test: Dict[int, Optional[Dataset]] = {}

        # self maintained task_id counter
        self.task_id: Optional[int] = None

        # data transformations
        self.base_transforms = {}
        self.normalize_transform = transforms.Normalize(MEAN, STD)


    @property
    def num_tasks(self) -> int:
        return self.hparams.num_tasks

    def classes(self, task_id: int) -> List[Any]:
        """Return class labels of task_id."""
        return [i for i in range(NUM_CLASSES)]

    def prepare_data(self):
        """Download data if needed."""
        OrigDataset(self.data_dir, train=True, download=True)
        OrigDataset(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data of self.task_id.

        Set variables here: self.data_train, self.data_val, self.data_test
        """
        # data transformations
        perm_seed = self.hparams.perm_seeds[self.task_id]
        permutation_transform = my_transforms.Permute(num_pixels=CHANNEL_SIZE, seed=perm_seed)
        self.base_transforms[self.task_id] = transforms.Compose([transforms.ToTensor(), permutation_transform])
        
        # target transformations
        one_hot_index = my_transforms.OneHotIndex(classes=self.classes(self.task_id))

        if stage == "fit":
            data_train_before_split = OrigDataset(
                root=self.data_dir,
                train=True,
                transform=transforms.Compose([self.base_transforms[self.task_id], self.normalize_transform]),
                target_transform=one_hot_index,
                download=False,
            )
            self.data_train, self.data_val = random_split(
                data_train_before_split,
                lengths=[1 - self.hparams.val_pc, self.hparams.val_pc],
                generator=torch.Generator().manual_seed(42),
            )
        elif stage == "test":
            self.data_test_orig[self.task_id] = OrigDataset(
                self.data_dir,
                train=False,
                transform=self.base_transforms[self.task_id],
                target_transform=one_hot_index,
                download=False,
            )
            
            self.data_test[self.task_id] = OrigDataset(
                self.data_dir,
                train=False,
                transform=transforms.Compose([self.base_transforms[self.task_id], self.normalize_transform]),
                target_transform=one_hot_index,
                download=False,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return {
            task_id: DataLoader(
                dataset=data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
            for task_id, data_test in self.data_test.items()
        }


if __name__ == "__main__":
    _ = PermutedCIFAR10()
