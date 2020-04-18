""" Parts of the GN-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtractFeatureMap(nn.Module):
        """1x1 convolution then upsampling and resize"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1x1 conv, output size: D x H/level X W/level if level else D x H X W
        self.1x1conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        x = self.1x1conv(x)
        # resize the feature to Dx(H*W)
        # H = x.size()[2]
        # W = x.size()[3]
        # # out_channels: D
        # x = x.view(out_channels, H*W)
        return x


        
