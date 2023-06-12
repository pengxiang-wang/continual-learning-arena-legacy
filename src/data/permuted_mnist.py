from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.data import transforms as my_transforms


class PermutedMNIST(LightningDataModule):
    """LightningDataModule for Pemuted MNIST dataset.

    TIL (Task-Incremental Learning) Setting. Must use HeadTIL for your model.

    Args:


    """

    def __init__(
        self,
        data_dir: str = "data/",
        num_tasks: int = 10,
        task_names: Optional[List[str]] = None,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        perm_seeds: List[int] = range(10),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.task_id = None
        self.task_names = task_names

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Dict[int, Optional[Dataset]] = {}

    @property
    def num_tasks(self) -> int:
        return self.hparams.num_tasks

    def classes(self, task_id: int) -> List[Any]:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def prepare_data(self):
        """Download data if needed."""
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data of self.task_id.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        # data transformations
        perm_seed = self.hparams.perm_seeds[self.task_id]
        permutation = my_transforms.Permute(num_pixels=784, seed=perm_seed)
        # target transformations
        one_hot_index = my_transforms.OneHotIndex(classes=self.classes(self.task_id))

        trainset = MNIST(
            root=self.hparams.data_dir,
            train=True,
            transform=transforms.Compose([self.transforms, permutation]),
            target_transform=one_hot_index,
        )
        testset = MNIST(
            self.hparams.data_dir,
            train=False,
            transform=transforms.Compose([self.transforms, permutation]),
            target_transform=one_hot_index,
        )
        dataset = ConcatDataset(datasets=[trainset, testset])
        data_train, data_val, data_test = random_split(
            dataset=dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        if stage == "fit":
            self.data_train = data_train
            self.data_val = data_val
        elif stage == "test":
            self.data_test[self.task_id] = data_test

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
    _ = PermutedMNIST()
