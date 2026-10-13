import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            channels = in_channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, 4 * growth_rate, 1, bias=False),
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False),
            ))

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            # Giữ đặc trưng cũ, nối thêm đặc trưng mới theo chiều kênh.
            x = torch.cat([x, new_features], dim=1)
        return x


def transition(in_channels):
    # Giảm một nửa số kênh và một nửa chiều cao, chiều rộng.
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
        nn.AvgPool2d(2),
    )


def build_densenet_manual():
    # DenseNet121-BC: 4 dense block gồm 6, 12, 24, 16 lớp; growth rate = 32.
    # Cấu trúc tham khảo bài báo: https://arxiv.org/abs/1608.06993
    channels = 64
    growth_rate = 32
    layers = [
        nn.Conv2d(1, channels, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2, padding=1),
    ]
    for block_index, num_layers in enumerate([6, 12, 24, 16]):
        layers.append(DenseBlock(channels, num_layers, growth_rate))
        channels += num_layers * growth_rate
        if block_index < 3:
            layers.append(transition(channels))
            channels //= 2

    layers.extend([
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(channels, 3),
    ])
    model = nn.Sequential(*layers)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.bias)
    return model
