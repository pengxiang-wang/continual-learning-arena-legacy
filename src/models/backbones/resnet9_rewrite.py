from torch import nn


class ResNet9(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.max_pool1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.max_pool2 = nn.MaxPool2d(4)

        self.flatten = nn.Flatten()

    def forward(self, xb):

        layer1 = self.conv1(xb)
        layer1 = self.bn1(layer1)
        layer1 = self.relu(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.bn2(layer2)
        layer2 = self.relu(layer2)
        layer2 = self.max_pool1(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.bn3(layer3)
        layer3 = self.relu(layer3)

        layer4 = self.conv4(layer3)
        layer4 = self.bn4(layer4)
        layer4 = self.relu(layer4)

        residual1 = layer4 + layer2

        layer5 = self.conv5(residual1)
        layer5 = self.bn5(layer5)
        layer5 = self.relu(layer5)
        layer5 = self.max_pool1(layer5)

        layer6 = self.conv6(layer5)
        layer6 = self.bn6(layer6)
        layer6 = self.relu(layer6)
        layer6 = self.max_pool1(layer6)

        layer7 = self.conv7(layer6)
        layer7 = self.bn7(layer7)
        layer7 = self.relu(layer7)

        layer8 = self.conv8(layer7)
        layer8 = self.bn8(layer8)
        layer8 = self.relu(layer8)

        residual2 = layer6 + layer8

        residual2 = self.max_pool2(residual2)

        feature = self.flatten(residual2)

        return feature


if __name__ == "__main__":
    _ = ResNet()
