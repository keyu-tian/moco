import torch
# from torch.nn.modules.batchnorm import _BatchNorm

from torchvision.models import resnet50
from model.res_cifar import resnet18


def model_entry(model_name: str, num_classes: int, norm_layer, **kwargs) -> torch.nn.Module:
    return globals()[model_name](num_classes=num_classes, norm_layer=norm_layer, **kwargs)
