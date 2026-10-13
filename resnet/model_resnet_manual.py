import torch
from torch import nn


class BasicBlock(nn.Module):
    """Basic residual block: y = ReLU(F(x) + shortcut(x)).

    Giu dac trung cu bang skip connection, hoc them phan du F(x).
    Vi du: vao 64 kenh -> conv 3x3 giu 64 kenh -> cong truc tiep dau vao.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut: identity neu cung shape, else conv 1x1 projection.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Noi tat: cho gradient chay thang, tranh vanishing gradient.
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=3):
        super().__init__()
        self.in_channels = 64
        # Stem: conv 7x7 + maxpool (chuan ResNet; AdaptiveAvgPool cuoi
        # ho tro moi crop size: 64x64 smoke test hay 128x128/224x224 anh that).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_resnet_manual():
    # ResNet-18: 4 stage gom 2, 2, 2, 2 BasicBlock (~11M params).
    # Cau truc tham khao bai bao: https://arxiv.org/abs/1512.03385
    # Ban tu dung day du mot kenh, 3 dau ra, khong chi them mot skip connection.
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=1, num_classes=3)


def resnet18(in_channels=1, num_classes=3):
    # Giu ten cu cua nhom de notebook/script cu van chay.
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
