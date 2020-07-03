""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from network.unet_parts import *


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