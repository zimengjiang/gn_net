""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from network.unet_parts import *

class EmbeddingNet(nn.Module):
    def __init__(self, n_channels = 3, D = 16, bilinear=False):
        super(EmbeddingNet, self).__init__()
        self.n_channels = n_channels
        self.D = D
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.F1 = OutConv(512, D)
        self.up2 = Up(512, 256, bilinear)
        self.F2 = OutConv(256, D)
        self.up3 = Up(256, 128, bilinear)
        self.F3 = OutConv(128, D)
        self.up4 = Up(128, 64 * factor, bilinear) # gives output features
        self.F4 = OutConv(64, D)
        # self.outc = OutConv(64, D)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        f1 = self.F1(x) # D x H/8 x W/8
        x = self.up2(x, x3)
        f2 = self.F2(x) # D x H/4 x W/4
        x = self.up3(x, x2)
        f3 = self.F3(x) # D x H/2 x W/2
        x = self.up4(x, x1) # output features 64 x H x W
        f4 = self.F4(x) # D x H x W
        # output = f4
        return f1,f2,f3,f4
    
    
    def get_embeddings(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # output features 64 x H x W
        return x

class GNNet(nn.Module):
    def __init__(self, embedding_net):
        super(GNNet, self).__init__()
        self.embedding_net = embedding_net
    
    def forward(self, input1, input2):
        output1 = self.embedding_net(input1)
        output2 = self.embedding_net(input2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

        


