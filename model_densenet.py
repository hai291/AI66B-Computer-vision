import torch
import torch.nn as nn


def build_densenet(pretrained=True):
    from torchvision.models import DenseNet121_Weights, densenet121

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    rgb_conv = model.features.conv0
    model.features.conv0 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.features.conv0.weight.copy_(rgb_conv.weight.sum(dim=1, keepdim=True))
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model
