import torch
import torch.nn as nn


def build_resnet(pretrained=False):
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    if not pretrained:
        model = resnet18(weights=None, num_classes=3)
        model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight)
        return model
    model = resnet18(weights=weights)
    rgb_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(rgb_conv.weight.sum(dim=1, keepdim=True))
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model
