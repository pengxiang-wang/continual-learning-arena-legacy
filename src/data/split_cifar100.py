from typing import Any, Dict, List, Optional, Callable

import numpy as np

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from src.data import transforms as my_transforms
from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()

NUM_CLASSES = 100
INPUT_SIZE = (3, 32, 32)
CHANNEL_SIZE = 32 * 32
MEAN = (0.5074, 0.4867, 0.4411)
STD = (0.2011, 0.1987, 0.2025)

DEFAULT_CLASS_SPLIT = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
    [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
]


class SplitCIFAR100(LightningDataModule):
    """LightningDataModule for Split MNIST dataset.

    CIL (Class-Incremental) scenario or TIL (Task-Incremental) scenario. You can use HeadsTIL or HeadsCIL for your model.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        scenario: str = "CIL",
        class_split: List[List[Any]] = DEFAULT_CLASS_SPLIT,
        joint: bool = False,
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
        self.joint = joint


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Dict[int, Optional[Dataset]] = {}

        # self maintained task_id counter
        self.task_id: Optional[int] = None

        # data transformations
        self.normalize_transform = transforms.Normalize(MEAN, STD)

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
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data of self.task_id.

        Set variables here: self.data_train, self.data_val, self.data_test
        """
        # target transformations
        one_hot_index = my_transforms.OneHotIndex(classes=self.classes(self.task_id))

        if stage == "fit":
            data_train_full_before_split = (CIFAR100(
                root=self.data_dir,
                train=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(20),
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1
                        ),
                        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
                        transforms.ToTensor(),
                        self.normalize_transform,
                        transforms.RandomErasing(
                            p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False
                        ),
                    ]
                ),
                target_transform=one_hot_index,
                download=False,
            )
            if not self.joint
                else TaskLabeledCIFAR100(
                    task_id=self.task_id,
                    root=self.data_dir,
                    train=True,
                    transform=transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(20),
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1
                        ),
                        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
                        transforms.ToTensor(),
                        self.normalize_transform,
                        transforms.RandomErasing(
                            p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False
                        ),
                    ]
                ),
                    target_transform=one_hot_index,
                    download=False,
            ))
            data_train_before_split = self._get_class_subset(
                data_train_full_before_split, classes=self.class_split[self.task_id]
            )


            data_train, data_val = random_split(
                data_train_before_split,
                lengths=[1 - self.hparams.val_pc, self.hparams.val_pc],
                generator=torch.Generator().manual_seed(42),
            )
            
            if (not self.joint) or self.task_id == 0:
                self.data_train = data_train
                self.data_val = data_val
            else:
                self.data_train = ConcatDataset([self.data_train, data_train])
                self.data_val = ConcatDataset([self.data_val, data_val])


        elif stage == "test":
            data_test = CIFAR100(
                self.data_dir,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), self.normalize_transform]
                ),
                target_transform=one_hot_index,
                download=False,
            )
            self.data_test[self.task_id] = self._get_class_subset(
                data_test, classes=self.class_split[self.task_id]
            )

    def _get_class_subset(
        self,
        dataset: CIFAR100,
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
        idx = np.array(dataset.targets) == classes[0]

        for cls in classes[1:]:
            idx = (np.array(dataset.targets) == cls) | idx

        dataset.data = dataset.data[idx]
        dataset.targets = [
            element for element, select_flag in zip(dataset.targets, idx) if select_flag
        ]
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

class TaskLabeledCIFAR100(CIFAR100):
    def __init__(
        self,
        task_id: int,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.__class__.__name__ = CIFAR100.__name__

        super().__init__(root, train, transform, target_transform, download)
        self.task_label = task_id

    def __getitem__(self, index: int):

        x, y = super().__getitem__(index)
        return x, y, self.task_label


if __name__ == "__main__":
    _ = SplitCIFAR100()
