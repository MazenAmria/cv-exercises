"""Model for CIFAR10."""

import torch as th
from torch import nn


class ConvModel(nn.Module):
    def __init__(self, input_channels: int, num_filters: int, verbose: bool = False):
        """
        Model definition.

        Args:
            input_channels: Number of input channels, this is 3 for the RGB images in CIFAR10
            num_filters: Number of convolutional filters
        """
        super().__init__()
        self.verbose = verbose
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(input_channels, num_filters, 3, 2, 1)
        self.norm1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(num_filters, 2 * num_filters, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(2 * num_filters)

        self.pool = nn.AvgPool2d(16, 16)
        self.linear = nn.Linear(2 * num_filters, 10)

    def forward(self, x: th.Tensor):
        """
        Model forward pass.

        Args:
            x: Model input, shape [batch_size, in_c, in_h, in_w]

        Returns:
            Model output, shape [batch_size, num_classes]
        """
        if self.verbose:
            print(f"Input shape: {x.shape}")

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        if self.verbose:
            print(f"Shape after first layer: {x.shape}")
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        if self.verbose:
            print(f"Shape after second layer: {x.shape}")
        
        x = self.pool(x)
        
        if self.verbose:
            print(f"Shape after averagepool: {x.shape}")

        x = th.reshape(x, (-1, 2 * self.num_filters))
        
        if self.verbose:
            print(f"Shape after reshape: {x.shape}")

        x = self.linear(x)
        
        if self.verbose:
            print(f"Model output shape: {x.shape}")
        
        return x
