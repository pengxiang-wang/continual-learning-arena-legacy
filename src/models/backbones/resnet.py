from torch import nn


# 换成torchvision里面的跑跑看


class ResNet(nn.Module):
    def __init__(self, block, layer_nums, input_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
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
        self, block, num_residual_blocks, expansion, in_channels, out_channels, stride
    ):
        identity_downsample = None
        layers = []

        if stride != 1 or in_channels != out_channels * expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * expansion, kernel_size=1, stride=stride
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
        self, in_channels, out_channels, expansion, identity_downsample=None, stride=1
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
        self, in_channels, out_channels, expansion, identity_downsample=None, stride=1
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


if __name__ == "__main__":
    _ = ResNet()
