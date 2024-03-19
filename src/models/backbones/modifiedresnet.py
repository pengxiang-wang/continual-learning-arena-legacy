from torch import nn
from torchvision import models


class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        # Modify the first layer to accept CIFAR-10 sized images
        self.resnet18.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Modify the output layer to have 10 classes instead of 1000
        self.resnet18.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet18(x)


if __name__ == "__main__":
    _ = ModifiedResNet18()
