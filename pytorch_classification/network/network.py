from .resnet import *
from .ghostnet import *
from .resnet_cifar import *
from .darknet import *
from .hrnet import *
from .resnet_blurpool import *

# from .new_network import *


__all__ = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnet_cifar",
    "ghostnet",
    "Darknet",
    "hrnet18",
    "hrnet32",
    "hrnet40",
]
