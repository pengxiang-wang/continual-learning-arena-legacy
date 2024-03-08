import torch
from torch import nn


MASK_GATE = nn.Sigmoid()

class WeightMaskedResNet(nn.Module):
    def __init__(self, block, layer_nums, input_channels):
        super().__init__()

        self.te = nn.ModuleDict()  # task embeddings over features
        self.mask_gate = MASK_GATE
        self.test_mask = None

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.te["conv1"] = nn.Embedding(1, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if block == MaskedBasicBlockSmall:
            expansion = 1
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        elif block == MaskedBasicBlockLarge:
            expansion = 4
            in_channels = [64, 256, 512, 1024]
            out_channels = [64, 128, 256, 512]

        # ResNet layers
        self.conv2_x = self._make_layer(
            "conv2_x",
            block,
            layer_nums[0],
            expansion,
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            stride=1,
        )
        self.conv3_x = self._make_layer(
            "conv3_x",
            block,
            layer_nums[1],
            expansion,
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            stride=2,
        )
        self.conv4_x = self._make_layer(
            "conv4_x",
            block,
            layer_nums[2],
            expansion,
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            stride=2,
        )
        self.conv5_x = self._make_layer(
            "conv5_x",
            block,
            layer_nums[3],
            expansion,
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def mask(self, task_embedding: nn.Embedding, scalar: float):
        return self.mask_gate(torch.tensor(scalar) * task_embedding.weight)

    def set_test_mask(self, mask):
        """Set task mask before testing task."""
        self.test_mask = mask

    def forward(self, x, scalar: float, stage: str):
        mask_record = {}

        h = self.conv1(x)
        m = (
            self.mask(self.te["conv1"], scalar)
            if stage == "fit"
            else self.test_mask["conv1"]
        )
        mask_record["conv1"] = m
        m = m.view(1, -1, 1, 1)
        h = m * h

        if stage == "train":  # problem! don't apply batchnorm at test stage
            h = self.bn1(h)
        a = self.relu(h)
        a = self.maxpool(a)

        for block in self.conv2_x:
            a = block(a, scalar, stage, mask_record, self.test_mask)
        for block in self.conv3_x:
            a = block(a, scalar, stage, mask_record, self.test_mask)
        for block in self.conv4_x:
            a = block(a, scalar, stage, mask_record, self.test_mask)
        for block in self.conv5_x:
            a = block(a, scalar, stage, mask_record, self.test_mask)
            

        a = self.avgpool(a)
        a = a.reshape(a.shape[0], -1)

        return a, mask_record

    def _make_layer(
        self,
        prefix,
        block,
        num_residual_blocks,
        expansion,
        in_channels,
        out_channels,
        stride,
    ):
        identity_downsample = None
        layers = nn.ModuleList()

        if stride != 1 or in_channels != out_channels * expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * expansion, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels * expansion),
            )

        layers.append(
            block(
                self.te,
                f"{prefix}0",
                in_channels,
                out_channels,
                expansion,
                identity_downsample,
                stride,
            )
        )
        in_channels = out_channels * expansion

        for i in range(1, num_residual_blocks):
            layers.append(
                block(
                    self.te,
                    f"{prefix}{i}",
                    in_channels,
                    out_channels,
                    expansion,
                    identity_downsample=None,
                    stride=1,
                )
            )

        return layers


class MaskedBasicBlockSmall(nn.Module):
    """Basic residual block for ResNet18, ResNet34."""

    def __init__(
        self,
        te,
        prefix,
        in_channels,
        out_channels,
        expansion,
        identity_downsample=None,
        stride=1,
    ):
        super().__init__()
        self.te = te
        self.mask_gate = MASK_GATE
        self.prefix = prefix  # for forward

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.te[f"{prefix}conv1"] = nn.Embedding(1, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.te[f"{prefix}conv2"] = nn.Embedding(1, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def mask(self, task_embedding: nn.Embedding, scalar: float):
        return self.mask_gate(torch.tensor(scalar) * task_embedding.weight)
    
    def forward(self, x, scalar: float, stage: str, mask_record, test_mask):
        identity = x

        h = self.conv1(x)
        m = (
            self.mask(self.te[f"{self.prefix}conv1"], scalar)
            if stage == "fit"
            else test_mask[f"{self.prefix}conv1"]
        )
        mask_record[f"{self.prefix}conv1"] = m
        m = m.view(1,-1,1,1)
        h = m * h
        if stage == "train":  # problem! don't apply batchnorm at test stage
            h = self.bn1(h)
        a = self.relu(h)

        h = self.conv2(a)
        m = (
            self.mask(self.te[f"{self.prefix}conv2"], scalar)
            if stage == "fit"
            else test_mask[f"{self.prefix}conv2"]
        )
        mask_record[f"{self.prefix}conv2"] = m
        m = m.view(1,-1,1,1)
        h = m * h
        if stage == "train":  # problem! don't apply batchnorm at test stage
            h = self.bn2(h)
        

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        h = h + identity  # residual

        a = self.relu(h)
        return a

class MaskedBasicBlockLarge(nn.Module):
    """Basic residual block for ResNet50, ResNet101, ResNet152."""

    def __init__(
        self,
        te,
        prefix,
        in_channels,
        out_channels,
        expansion,
        identity_downsample=None,
        stride=1,
    ):
        super().__init__()
        self.te = te
        self.mask_gate = MASK_GATE
        self.prefix = prefix  # for forward

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.te[f"{prefix}conv1"] = nn.Embedding(1, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.te[f"{prefix}conv2"] = nn.Embedding(1, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.te[f"{prefix}conv3"] = nn.Embedding(1, out_channels * expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def mask(self, task_embedding: nn.Embedding, scalar: float):
        return self.mask_gate(torch.tensor(scalar) * task_embedding.weight)

    def forward(self, x, scalar: float, stage: str, mask_record, test_mask):
        identity = x

        h = self.conv1(x)
        m = (
            self.mask(self.te[f"{self.prefix}conv1"], scalar)
            if stage == "fit"
            else test_mask[f"{self.prefix}conv1"]
        )
        mask_record[f"{self.prefix}conv1"] = m
        m = m.view(1,-1,1,1)
        h = m * h
        if stage == "train":  # problem! don't apply batchnorm at test stage
            h = self.bn1(h)
        
        a = self.relu(h)

        h = self.conv2(a)
        m = (
            self.mask(self.te[f"{self.prefix}conv2"], scalar)
            if stage == "fit"
            else test_mask[f"{self.prefix}conv2"]
        )
        mask_record[f"{self.prefix}conv2"] = m
        m = m.view(1,-1,1,1)
        h = m * h
        if stage == "train":  # problem! don't apply batchnorm at test stage
            h = self.bn2(h)
        a = self.relu(h)

        h = self.conv3(a)
        m = (
            self.mask(self.te[f"{self.prefix}conv3"], scalar)
            if stage == "fit"
            else test_mask[f"{self.prefix}conv3"]
        )
        mask_record[f"{self.prefix}conv3"] = m
        m = m.view(1,-1,1,1)
        h = m * h
        if stage == "train":  # problem! don't apply batchnorm at test stage
            h = self.bn3(h)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        h = h + identity

        a = self.relu(h)
        return a


class MaskedResNet18(MaskedResNet):
    def __init__(self, input_channels):
        super().__init__(MaskedBasicBlockSmall, [2, 2, 2, 2], input_channels)


class MaskedResNet34(MaskedResNet):
    def __init__(self, input_channels):
        super().__init__(MaskedBasicBlockSmall, [3, 4, 6, 3], input_channels)


class MaskedResNet50(MaskedResNet):
    def __init__(self, input_channels):
        super().__init__(MaskedBasicBlockLarge, [3, 4, 6, 3], input_channels)


class MaskedResNet101(MaskedResNet):
    def __init__(self, input_channels):
        super().__init__(MaskedBasicBlockLarge, [3, 4, 23, 3], input_channels)


class MaskedResNet152(MaskedResNet):
    def __init__(self, input_channels):
        super().__init__(MaskedBasicBlockLarge, [3, 8, 36, 3], input_channels)


if __name__ == "__main__":
    _ = MaskedResNet()
