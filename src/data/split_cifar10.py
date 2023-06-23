from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.data import transforms as my_transforms
from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_loggerpack()



NUM_CLASSES = 10
INPUT_SIZE = (3,32,32)
INPUT_LEN = 3*32*32
MEAN = (0.4914, 0.4822 ,0.4465)
STD = (0.2470, 0.2435, 0.2616)

DEFAULT_CLASS_SPLIT = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


class SplitCIFAR10(LightningDataModule):
    """LightningDataModule for Split MNIST dataset.

    CIL (Class-Incremental) scenario or TIL (Task-Incremental) scenario. You can use HeadTIL or HeadCIL for your model.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        scenario: str = "CIL",
        class_split: List[List[Any]] = DEFAULT_CLASS_SPLIT,
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
        self.class_split = class_split

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Dict[int, Optional[Dataset]] = {}

        # self maintained task_id counter
        self.task_id: Optional[int] = None

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    @property
    def num_tasks(self) -> int:
        return len(self.class_split)

    def classes(self, task_id: int) -> List[Any]:
        """Return class labels of task_id."""

        if self.scenario == "CIL":
            classes = []
            for t in range(task_id + 1):
                classes.extend(
                    self.class_split[t]
                )  # classes grows incrementally for CIL scenario
        elif self.scenario == "TIL":
            classes = self.class_split[task_id]
        return classes

    def prepare_data(self):
        """Download data if needed."""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data of self.task_id.

        Set variables here: self.data_train, self.data_val, self.data_test
        """
        # target transformations
        one_hot_index = my_transforms.OneHotIndex(classes=self.classes(self.task_id))

        if stage == "fit":
            data_train_full_before_split = CIFAR10(
                root=self.data_dir,
                train=True,
                transform=transforms.Compose([self.transforms]),
                target_transform=one_hot_index,
                download=False,
            )
            data_train_before_split = self._get_class_subset(
                data_train_full_before_split, classes=self.class_split[self.task_id]
            )
            self.data_train, self.data_val = random_split(
                data_train_before_split,
                lengths=[1 - self.hparams.val_pc, self.hparams.val_pc],
                generator=torch.Generator().manual_seed(42),
            )
        elif stage == "test":
            data_test = CIFAR10(
                self.data_dir,
                train=False,
                transform=transforms.Compose([self.transforms]),
                target_transform=one_hot_index,
                download=False,
            )
            self.data_test[self.task_id] = self._get_class_subset(
                data_test, classes=self.class_split[self.task_id]
            )

    def _get_class_subset(
        self,
        dataset: CIFAR10,
        classes: list[int],
    ) -> Dataset:
        """Get subset of dataset in certain classes, as an implementation for spliting dataset.

        Args:
            dataset (nn.Dataset): original dataset to be retrieved.
            classes (list[int]): retrieved class.

        Returns:
            nn.Dataset: subset of original dataset in classes.
        """
        # Get from dataset.data and dataset.targets
        idx = dataset.targets == classes[0]
        for cls in classes[1:]:
            idx = (dataset.targets == cls) | idx
        dataset.data = dataset.data[idx]
        dataset.targets = dataset.targets[idx]
        return dataset

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
    _ = SplitCIFAR10()
