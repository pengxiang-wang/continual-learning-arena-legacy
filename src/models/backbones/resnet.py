from torch import nn


class BasicBlockSmall(nn.Module):
    """Basic residual block for ResNet18, ResNet34."""

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
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

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity  # residual
        x = self.relu(x)
        return x


class BasicBlockLarge(nn.Module):
    """Basic residual block for ResNet50, ResNet101, ResNet152."""

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
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
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layer_nums, input_channels):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.conv2_x = self._make_layer(block, layer_nums[0], out_channel=64, stride=1)
        self.conv3_x = self._make_layer(block, layer_nums[1], out_channel=128, stride=2)
        self.conv4_x = self._make_layer(block, layer_nums[2], out_channel=256, stride=2)
        self.conv5_x = self._make_layer(block, layer_nums[3], out_channel=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # output feature 2048

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * 4, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    identity_downsample=None,
                    stride=stride,
                )
            )

        return nn.Sequential(*layers)


class ResNet18(ResNet):
    def __init__(self, input_channels=3):
        super().__init__(BasicBlockSmall, [2, 2, 2, 2], input_channels)


class ResNet34(ResNet):
    def __init__(self, input_channels=3):
        super().__init__(BasicBlockSmall, [3, 4, 6, 3], input_channels)


class ResNet50(ResNet):
    def __init__(self, input_channels=3):
        super().__init__(BasicBlockLarge, [3, 4, 6, 3], input_channels)


class ResNet101(ResNet):
    def __init__(self, input_channels=3):
        super().__init__(BasicBlockLarge, [3, 4, 23, 3], input_channels)


class ResNet152(ResNet):
    def __init__(self, input_channels=3):
        super().__init__(BasicBlockLarge, [3, 8, 36, 3], input_channels)


if __name__ == "__main__":
    _ = ResNet()
