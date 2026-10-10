import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Khoi residual co ban: y = F(x) + shortcut(x)."""

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

        # Nhanh tat: noi thang x neu cung kich thuoc, khong thi chieu qua conv 1x1
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
        out = out + self.shortcut(x)  # skip connection: cho gradient chay thang
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=3):
        super().__init__()
        self.in_channels = 64
        # Stem: conv 7x7 + maxpool (chuan ResNet, AdaptiveAvgPool cuoi giup
        # nhan moi kich thuoc crop: 64x64 demo hay 224x224 anh that)
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


def resnet18(in_channels=1, num_classes=3):
    """ResNet-18: [2, 2, 2, 2] BasicBlock, ~11M params — nhe nhat, hop dataset y te nho."""
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for size in (64, 224):
        model = resnet18(in_channels=1, num_classes=3)
        out = model(torch.randn(2, 1, size, size))
        print(f"input {size}x{size} -> output {tuple(out.shape)}")

    print(f"ResNet-18 params: {count_parameters(resnet18()):,}")

    from model import LumbarCNN  # baseline de so sanh
    print(f"LumbarCNN baseline params: {count_parameters(LumbarCNN()):,}")
