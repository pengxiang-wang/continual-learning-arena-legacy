import torch


class Permute:
    """Permute transform for image.

    Used as a PyTorch Dataset Transform.

    Args:
        seed (int): permutation seed
    """

    def __init__(self, num_pixels: int, seed: int = None):
        self.torch_generator = torch.Generator()
        if seed:
            self.torch_generator.manual_seed(seed)

        self.idx = torch.randperm(num_pixels, generator=self.torch_generator)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): MNIST image to be permuted.

        Returns:
            torch.Tensor: permuted image.
        """
        orig_size = img.size()
        img = img.view(-1)
        img = img[self.idx].view(orig_size)
        return img


if __name__ == "__main__":
    permutation = Permute()
    from torchvision import datasets, transforms

    orig_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=False
    )
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                permutation,
            ]
        ),
        download=False,
    )
    print(orig_dataset.data)
    print(dataset.data)
    # target_transform is only applied during indexing like dataloader.
    from torch.utils.data import DataLoader

    orig_dataloader = DataLoader(orig_dataset, batch_size=64, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    from matplotlib import pyplot as plt

    for x, y in orig_dataloader:
        x0 = x[0].squeeze()
        print(x0)
        plt.imshow(x0)
        break
    plt.show()
    for x, y in dataloader:
        x0 = x[0].squeeze()
        print(x0)
        plt.imshow(x0)
        break
    plt.show()
