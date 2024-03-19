import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing

# from torchvision.models.resnet18 import ResNet18
import torchvision.models as models

mps = False

if mps:
    mps_device = torch.device("gpu")


def main():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

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

    class ResNet(nn.Module):
        def __init__(self, block, layer_nums, input_channels):
            super().__init__()

            self.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            if block == BasicBlockSmall:
                expansion = 1
                in_channels = [64, 64, 128, 256]
                out_channels = [64, 128, 256, 512]
            elif block == BasicBlockLarge:
                expansion = 4
                in_channels = [64, 256, 512, 1024]
                out_channels = [64, 128, 256, 512]

            # ResNet layers
            self.conv2_x = self._make_layer(
                block,
                layer_nums[0],
                expansion,
                in_channels=in_channels[0],
                out_channels=out_channels[0],
                stride=1,
            )
            self.conv3_x = self._make_layer(
                block,
                layer_nums[1],
                expansion,
                in_channels=in_channels[1],
                out_channels=out_channels[1],
                stride=2,
            )
            self.conv4_x = self._make_layer(
                block,
                layer_nums[2],
                expansion,
                in_channels=in_channels[2],
                out_channels=out_channels[2],
                stride=2,
            )
            self.conv5_x = self._make_layer(
                block,
                layer_nums[3],
                expansion,
                in_channels=in_channels[3],
                out_channels=out_channels[3],
                stride=2,
            )

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            h = self.conv1(x)
            h = self.bn1(h)
            a = self.relu(h)
            a = self.maxpool(a)

            a = self.conv2_x(a)
            a = self.conv3_x(a)
            a = self.conv4_x(a)
            a = self.conv5_x(a)

            a = self.avgpool(a)
            a = a.reshape(a.shape[0], -1)

            return a

        def _make_layer(
            self,
            block,
            num_residual_blocks,
            expansion,
            in_channels,
            out_channels,
            stride,
        ):
            identity_downsample = None
            layers = []

            if stride != 1 or in_channels != out_channels * expansion:
                identity_downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels * expansion,
                        kernel_size=1,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(out_channels * expansion),
                )

            layers.append(
                block(in_channels, out_channels, expansion, identity_downsample, stride)
            )
            in_channels = out_channels * expansion

            for i in range(num_residual_blocks - 1):
                layers.append(
                    block(
                        in_channels,
                        out_channels,
                        expansion,
                        identity_downsample=None,
                        stride=1,
                    )
                )

            return nn.Sequential(*layers)

    class BasicBlockSmall(nn.Module):
        """Basic residual block for ResNet18, ResNet34."""

        def __init__(
            self,
            in_channels,
            out_channels,
            expansion,
            identity_downsample=None,
            stride=1,
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.identity_downsample = identity_downsample

        def forward(self, x):
            identity = x

            h = self.conv1(x)
            h = self.bn1(h)
            a = self.relu(h)

            h = self.conv2(a)
            h = self.bn2(h)

            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)
            h = h + identity  # residual

            a = self.relu(h)
            return a

    class BasicBlockLarge(nn.Module):
        """Basic residual block for ResNet50, ResNet101, ResNet152."""

        def __init__(
            self,
            in_channels,
            out_channels,
            expansion,
            identity_downsample=None,
            stride=1,
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(
                out_channels,
                out_channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.bn3 = nn.BatchNorm2d(out_channels * expansion)
            self.relu = nn.ReLU()
            self.identity_downsample = identity_downsample

        def forward(self, x):
            identity = x

            h = self.conv1(x)
            h = self.bn1(h)
            a = self.relu(h)

            h = self.conv2(a)
            h = self.bn2(h)
            a = self.relu(h)

            h = self.conv3(a)
            h = self.bn3(h)

            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)
            h = h + identity

            a = self.relu(h)
            return a

    class ResNet18(ResNet):
        def __init__(self, input_channels):
            super().__init__(BasicBlockSmall, [2, 2, 2, 2], input_channels)

    class ResNet34(ResNet):
        def __init__(self, input_channels):
            super().__init__(BasicBlockSmall, [3, 4, 6, 3], input_channels)

    class ResNet50(ResNet):
        def __init__(self, input_channels):
            super().__init__(BasicBlockLarge, [3, 4, 6, 3], input_channels)

    class ResNet101(ResNet):
        def __init__(self, input_channels):
            super().__init__(BasicBlockLarge, [3, 4, 23, 3], input_channels)

    class ResNet152(ResNet):
        def __init__(self, input_channels):
            super().__init__(BasicBlockLarge, [3, 8, 36, 3], input_channels)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool1 = nn.AvgPool2d(2, 2)
            self.norm1 = nn.BatchNorm2d(6)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pool2 = nn.AvgPool2d(2, 2)
            self.norm2 = nn.BatchNorm2d(16)
            self.fc1 = nn.Linear(16 * 5 * 5, 10)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)

            x = self.fc1(x)
            return x

    net = ResNet18(3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if mps:
        net = net.to(mps_device)
        criterion = criterion.to(mps_device)

    for epoch in range(50):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if mps:
                inputs = inputs.to(mps_device)
                labels = labels.to(mps_device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        print("Epoch %d, Training Accuracy: %.2f %%" % (epoch + 1, train_accuracy))
        print(
            "Epoch %d, Training Loss: %.3f"
            % (epoch + 1, running_loss / len(trainloader))
        )

        correct_test = 0
        total_test = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if mps:
                    images = images.to(mps_device)
                    labels = labels.to(mps_device)

                outputs = net(images)
                _, predicted_test = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        print("Epoch %d, Testing Accuracy: %.2f %%" % (epoch + 1, test_accuracy))

    print("Finished Training")


if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()
