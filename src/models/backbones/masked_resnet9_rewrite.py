import torch
from torch import nn



class MaskedResNet9(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.te = nn.ModuleDict()  # task embeddings over features
        self.mask_gate = nn.Sigmoid()
        self.test_mask = None

        self.max_pool1 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.te["conv1"] = nn.Embedding(1, 64)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.te["conv2"] = nn.Embedding(1, 128)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.te["conv3"] = nn.Embedding(1, 128)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.te["conv4"] = nn.Embedding(1, 128)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.te["conv5"] = nn.Embedding(1, 256)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.te["conv6"] = nn.Embedding(1, 512)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.te["conv7"] = nn.Embedding(1, 512)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.te["conv8"] = nn.Embedding(1, 512)
        self.bn8 = nn.BatchNorm2d(512)

        self.max_pool2 = nn.MaxPool2d(4)

        self.flatten = nn.Flatten()

        self.test_mask = None

        self.module_order = [f"conv{l}" for l in range(1,9)]# ,5)] + ["residual"] + [f"conv{l}" for l in range(7, 9)]

    def mask(self, task_embedding: nn.Embedding, scalar: float):
        return self.mask_gate(torch.tensor(scalar) * task_embedding.weight)

    def set_test_mask(self, mask):
        """Set task mask before testing task."""
        self.test_mask = mask

    def forward(self, x, scalar: float, stage: str):

        mask_record = {}  # for mask regularisaion terms

        layer1 = self.conv1(x)
        m = (
            self.mask(self.te["conv1"], scalar)
            if stage == "fit"
            else self.test_mask["conv1"]
        )
        mask_record["conv1"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer1 = m * layer1
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer1 = self.bn1(layer1)
        layer1 = self.relu(layer1)
        
        layer2 = self.conv2(layer1)
        m = (
            self.mask(self.te["conv2"], scalar)
            if stage == "fit"
            else self.test_mask["conv2"]
        )
        mask_record["conv2"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer2 = m * layer2
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer2 = self.bn2(layer2)
        layer2 = self.relu(layer2)
        layer2 = self.max_pool1(layer2)

        layer3 = self.conv3(layer2)
        m = (
            self.mask(self.te["conv3"], scalar)
            if stage == "fit"
            else self.test_mask["conv3"]
        )
        mask_record["conv3"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer3 = m * layer3
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer3 = self.bn3(layer3)
        layer3 = self.relu(layer3)

        layer4 = self.conv4(layer3)
        m = (
            self.mask(self.te["conv4"], scalar)
            if stage == "fit"
            else self.test_mask["conv4"]
        )
        mask_record["conv4"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer4 = m * layer4
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer4 = self.bn4(layer4)
        layer4 = self.relu(layer4)

        residual1 = layer4 + layer2
        # residual1 = self.relu(residual1)

        layer5 = self.conv5(residual1)
        m = (
            self.mask(self.te["conv5"], scalar)
            if stage == "fit"
            else self.test_mask["conv5"]
        )
        mask_record["conv5"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer5 = m * layer5
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer5 = self.bn5(layer5)
        layer5 = self.relu(layer5)
        layer5 = self.max_pool1(layer5)

        layer6 = self.conv6(layer5)
        m = (
            self.mask(self.te["conv6"], scalar)
            if stage == "fit"
            else self.test_mask["conv6"]
        )
        mask_record["conv6"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer6 = m * layer6
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer6 = self.bn6(layer6)
        layer6 = self.relu(layer6)
        layer6 = self.max_pool1(layer6)

        layer7 = self.conv7(layer6)
        m = (
            self.mask(self.te["conv7"], scalar)
            if stage == "fit"
            else self.test_mask["conv7"]
        )
        mask_record["conv7"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer7 = m * layer7
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer7 = self.bn7(layer7)
        layer7 = self.relu(layer7)

        layer8 = self.conv8(layer7)
        m = (
            self.mask(self.te["conv8"], scalar)
            if stage == "fit"
            else self.test_mask["conv8"]
        )
        mask_record["conv8"] = m  # for mask regularisaion terms
        m = m.view(1, -1, 1, 1)
        layer8 = m * layer8
        if stage == "train":  # problem! don't apply batchnorm at test stage
            layer8 = self.bn8(layer8)
        layer8 = self.relu(layer8)

        residual2 = layer8 + layer6
        # residual2 = self.relu(residual2)


        residual2 = self.max_pool2(residual2)

        feature = residual2.reshape(residual2.shape[0], -1)

        return feature, mask_record


if __name__ == "__main__":
    _ = MaskedResNet9(3)
