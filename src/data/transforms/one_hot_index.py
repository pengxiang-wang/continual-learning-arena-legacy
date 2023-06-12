from typing import Any, Dict, List

import torch


class OneHotIndex:
    """Permute transform for MNIST image.

    Used as a PyTorch Dataset Transform.

    Args:
        seed (int): permutation seed
    """

    def __init__(self, classes: List[Any]):
        self.class_indices = {cls: classes.index(cls) for cls in classes}

    def __call__(self, target: Any) -> torch.Tensor:
        """
        Args:
            target: original label.

        Returns:
            torch.Tensor:
        """
        target = self.class_indices[target]
        return target


if __name__ == "__main__":
    one_hot_index = OneHotIndex([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    from torchvision import datasets, transforms

    orig_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=False
    )
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        target_transform=one_hot_index,
        download=False,
    )
    print(orig_dataset.targets)
    print(dataset.targets)
    # target_transform is only applied during indexing like dataloader.
    from torch.utils.data import DataLoader

    orig_dataloader = DataLoader(orig_dataset, batch_size=64, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    for x, y in orig_dataloader:
        print(y)
        break
    for x, y in dataloader:
        print(y)
        break
