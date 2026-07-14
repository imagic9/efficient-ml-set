"""VGG11 taken from torchvision and adapted to CIFAR-10 (32x32, 10 classes)."""
import torch.nn as nn
from torchvision.models import vgg11, vgg11_bn


def build_vgg11_cifar(num_classes: int = 10, batch_norm: bool = True):
    """VGG11 convolutional backbone from torchvision with a CIFAR-sized head.

    torchvision's VGG11 targets 224x224 ImageNet: after the 5 max-pools a 224
    input becomes 7x7 and the classifier is three huge FC layers (~124M params).
    A 32x32 CIFAR input collapses to 1x1 after the same backbone, so we swap the
    ImageNet head for a compact classifier. We keep the exact torchvision feature
    extractor, so this is still "VGG11 from torchvision", just re-headed for the
    dataset. batch_norm=True uses vgg11_bn, which trains far more reliably from
    scratch on CIFAR.
    """
    net = vgg11_bn(weights=None) if batch_norm else vgg11(weights=None)
    net.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 1x1 spatial output -> 512 features
    net.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return net


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
