"""Assemble image retrieval network, with intermediate endpoints.
"""
import gin
from collections import OrderedDict
# from network.images_from_list import ImagesFromList
from network.netvlad import NetVLAD
from tqdm import tqdm
from typing import List
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.functional import interpolate
from torchvision import models

class MyImageRetrievalModel(nn.Module):
    """Build the image retrieval model with intermediate feature extraction.

    The model is made of a VGG-16 backbone combined with a NetVLAD pooling
    layer.
    """
    def __init__(self):
        """Initialize the Image Retrieval Network.

        Args:
            checkpoint_path: Path to the pre-trained weights.
            hypercolumn_layers: The hypercolumn layer indices used to compute
                the intermediate features.
            device: The pytorch device to run on.
        """
        super(MyImageRetrievalModel, self).__init__()
        self._hypercolumn_layers = [14, 17, 21, 24, 28] # use the same layers as S2DHM
        encoder = models.vgg16(pretrained=False)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        self._model = encoder

    def forward(self, x):
        '''x is the input image tensor'''
        feature_maps,j = [],0
        for i, layer in enumerate(list(self._model.children())):
                if(j==len(self._hypercolumn_layers)):
                    break
                if(i==self._hypercolumn_layers[j]):
                    feature_maps.append(x)
                    j+=1
                x = layer(x) # forwarding
        # Delete and empty cache
        del x
        torch.cuda.empty_cache()
        return feature_maps
