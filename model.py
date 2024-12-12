from signal import pause
from torch import nn

import ai8x

import matplotlib
import matplotlib.pyplot as plt

"""
Network description class
"""
class FaceNet(nn.Module):
    """
    7-Layer CNN - Lightweight image classification
    """
    def __init__(self, num_classes=2, dimensions=(88, 88), num_channels=3, bias=False, **kwargs):
        super().__init__()

        # Keep track of image dimensions so one constructor works for all image sizes
        dim_x, dim_y = dimensions
        assert dim_x == dim_y == 88 # Only square supported

        self.conv1 = ai8x.FusedConv2dReLU(in_channels=num_channels, out_channels=4, kernel_size=3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions

        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels=4, out_channels=8, kernel_size=3,
                                                 padding=1, bias=bias, **kwargs)
        # conv padding 1 -> no change in dimensions 
        # pooling, padding 0 -> dimensions halved
        dim_x //= 2 # 44
        dim_y //= 2 # 44

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(in_channels=8, out_channels=16, kernel_size=3,
                                                 padding=1, bias=bias, **kwargs)
        # pooling, padding 0 -> dimensions halved
        dim_x //= 2 # 22
        dim_y //= 2 # 22

        self.conv4 = ai8x.FusedAvgPoolConv2dReLU(in_channels=16, out_channels=32, kernel_size=3,
                                                 padding=1, bias=bias, **kwargs)
        # conv padding 1 -> no change in dimensions
        # pooling, padding 0 -> dimensions halved
        dim_x //= 2 # 11
        dim_y //= 2 # 11

        self.conv5 = ai8x.FusedAvgPoolConv2dReLU(in_channels=32, out_channels=32, kernel_size=3,
                                                 padding=1, bias=bias, **kwargs)
        # conv padding 1 -> no change in dimensions
        # pooling, padding 0 -> dimensions halved
        dim_x //= 2 # 5
        dim_y //= 2 # 5

        self.fcx = ai8x.Linear(dim_x*dim_y*32, num_classes, wide=True, bias=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fcx(x)
        return x


def facenet(pretrained=False, **kwargs):
    """
    Constructs a MemeNet model.
    """
    assert not pretrained
    return FaceNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'facenet',
        'min_input': 1,
        'dim': 2,
    }
]
