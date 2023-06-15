import copy
from indexed import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.data import transforms as my_transforms

# Unfinished


class SplitMNIST(LightningDataModule):
    """LightningDataModule for Split MNIST dataset.

    CIL (Class-Incremental) scenario or TIL (Task-Incremental) scenario. You can use HeadTIL or HeadCIL for your model.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        class_split: List[List[Any]] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: OrderedDict[str, Optional[Dataset]] = OrderedDict()

    @property
    def num_tasks(self):
        return len(self.class_split)

    def classes(self, task_id):
        classes = []
        for t in range(task_id):
            classes.extend(self.class_split[t])
        return classes

    def prepare_data(self):
        """Download data if needed."""
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data of self.task_id.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        # target transformations
        one_hot_index = my_transforms.OneHotIndex(classes=self.classes(self.task_id))

        trainset = MNIST(
            self.hparams.data_dir,
            train=True,
            transform=self.transforms,
            target_transform=one_hot_index,
            download=False,
        )
        testset = MNIST(
            self.hparams.data_dir,
            train=False,
            transform=self.transforms,
            target_transform=one_hot_index,
            download=False,
        )
        dataset = ConcatDataset(datasets=[trainset, testset])

        subset = self._get_class_subset(dataset, self.class_split[self.task_id])
        train_val_test_split_each_task = [
            int(pc * len(subset)) for pc in self.hparams.train_val_test_split_each_task
        ]
        data_train, data_val, data_test = random_split(
            dataset=subset,
            lengths=train_val_test_split_each_task,
            generator=torch.Generator().manual_seed(42),
        )
        if stage == "fit":
            self.data_train = data_train
            self.data_val = data_val
        elif stage == "test":
            self.data_test[self.task_id] = data_test

    def _get_class_subset(
        self, dataset: Dataset, classes: list[int]
    ) -> torch.nn.Dataset:
        """Get subset of dataset in certain classes, as an implementation for spliting dataset.

        Args:
            dataset (nn.Dataset): original dataset to be retrieved.
            classes (list[int]): retrieved class.

        Returns:
            nn.Dataset: subset of original dataset in classes.
        """
        # Get from dataset.data and dataset.targets
        subset = copy.deepcopy(dataset)
        subset.data = []
        subset.targets = []
        for x, y in dataset:
            if y.item() in classes:
                subset.data.append(x)
                subset.targets.append(y)
        # Reconstruct subset.data, subset.targets
        subset.data = torch.cat(subset.data)
        subset.targets = torch.cat(subset.targets)
        return subset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test[self.task_id_test],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Test dataloader of other task."""
        return [
            DataLoader(
                dataset=data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
            for data_test in self.data_test
        ]


if __name__ == "__main__":
    _ = SplitMNIST()
