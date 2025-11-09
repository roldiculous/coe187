###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Rock-Paper-Scissors classification network for AI85
"""

# Essential Imports
from torch import nn
import ai8x


class AI85RPSNet(nn.Module):
    """
    Define CNN model for image classification.
    """
    def __init__(self, num_classes=3, num_channels=3, dimensions=(128, 128),
                 fc_inputs=16, bias=False, **kwargs):
        super().__init__()

        # AI85 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes_
        dim = dimensions[0]
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1, bias=bias, **kwargs)
        pad = 2 if dim == 28 else 1

        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 32, 3, pool_size=2, pool_stride=2, padding=pad, bias=bias, **kwargs)
        dim //= 2
        if pad == 2:
            dim += 2

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        dim //= 2

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(64, 32, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        dim //= 2

        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        dim //= 2

        self.conv6 = ai8x.FusedConv2dReLU(32, fc_inputs, 3, padding=1, bias=bias, **kwargs)

        self.fc = ai8x.Linear(fc_inputs*dim*dim, num_classes, bias=True, wide=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai85rpsnet(pretrained=False, **kwargs):
    """
    Constructs a AI85RPSNet model.
    """
    assert not pretrained
    return AI85RPSNet(**kwargs)


models = [
    {
        'name': 'ai85rpsnet',
        'min_input': 1,
        'dim': 2,
    },
]
