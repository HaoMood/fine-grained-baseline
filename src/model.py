# -*- coding: utf-8 -*-
"""Define the convolution neural network model.

Tricks:
    - Use batch harmonic to rescale the GAP activations.
"""


import collections

import torch
import torchvision


torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_num_threads(32)


__all__ = ['Model']
__author__ = 'Hao Zhang'
__copyright__ = '2019 LAMDA'
__date__ = '2019-04-05'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 4.0'
__status__ = 'Development'
__updated__ = '2019-04-15'
__version__ = '2.4'


class Model(torch.nn.Module):
    """The convolution neural network model.

    Attributes:
        architecture: str. Which backbone model to use.
        input_size: int. Input image size.
        num_classes: int.
        phase: str = extract/fc/all. Training fc or all layers.
        backbone: torch.nn.Module.
        gap: torch.nn.Module.
        bh: torch.nn.Module.
        fc: torch.nn.Module.
    """
    def __init__(self, architecture: str, input_size: int, num_classes: int):
        """Network initialization."""
        torch.nn.Module.__init__(self)
        self.architecture = architecture
        self.input_size = input_size
        self.num_classes = num_classes
        self.phase = None

        if self.architecture.startswith('vgg'):
            backbone = getattr(torchvision.models,
                               self.architecture)(pretrained=True)
            self.backbone = backbone.features[:-1]
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(
                in_features=self.backbone[-2].out_channels,
                out_features=num_classes, bias=True)
        elif self.architecture.startswith('resnet'):
            backbone = getattr(torchvision.models,
                               self.architecture)(pretrained=True)
            self.backbone = torch.nn.Sequential(collections.OrderedDict(
                list(backbone.named_children())[:-2]))
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(
                in_features=backbone.layer4[-1].bn2.num_features,
                out_features=num_classes, bias=True)
        else:
            raise NotImplementedError

        self._initParameters(self.fc)

    def _initParameters(self, layer: torch.nn.Module):
        """Initialize the weight and bias for the given layer."""
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                          nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        N = X.size()[0]
        assert X.size() == (N, 3, self.input_size, self.input_size)
        # Forward backbone network.
        X = self.backbone(X)

        # GAP and reshape
        X = self.gap(X)
        X = torch.reshape(X, (N, -1))

        # Classification.
        X = self.fc(X)
        assert X.size() == (N, self.num_classes)
        return X
