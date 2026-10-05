import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class MBConv(nn.Module):
    """Mobile inverted bottleneck block used by EfficientNet."""

    def __init__(self, in_channels, out_channels, expand_ratio, stride, drop_rate=0.0):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_path = nn.Identity() if drop_rate == 0 else StochasticDepth(drop_rate)

        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(expanded_channels),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True),
                SqueezeExcitation(expanded_channels),
                nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result = x + self.drop_path(result)
        return result


class StochasticDepth(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        keep_rate = 1 - self.drop_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_rate + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_rate) * random_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class LumbarCNN(nn.Module):
    """CNN cơ bản: nhận 1 ảnh crop MRI, trả về logits mức độ nghiêm trọng (Normal/Mild, Moderate, Severe)."""

    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 classifier for grayscale MRI crops."""

    def __init__(self, in_channels=1, num_classes=3, drop_rate=0.2, drop_path_rate=0.2):
        super().__init__()
        settings = (
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        total_blocks = sum(blocks for _, _, blocks, _ in settings)
        block_index = 0
        features = []
        in_features = 32
        for expand_ratio, out_features, blocks, stride in settings:
            for block in range(blocks):
                current_stride = stride if block == 0 else 1
                drop_rate_for_block = drop_path_rate * block_index / max(1, total_blocks - 1)
                features.append(
                    MBConv(
                        in_features,
                        out_features,
                        expand_ratio,
                        current_stride,
                        drop_rate_for_block,
                    )
                )
                in_features = out_features
                block_index += 1
        self.features = nn.Sequential(*features)
        self.head = nn.Sequential(
            nn.Conv2d(in_features, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    model = EfficientNetB0()
    x = torch.randn(4, 1, 128, 128)
    out = model(x)
    print(out.shape)
