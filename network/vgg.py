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

class ImageRetrievalModel(nn.Module):
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
        super(ImageRetrievalModel, self).__init__()
        self._hypercolumn_layers = [14, 17, 21, 24, 28]
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
        

    # def compute_hypercolumn(self, image: List[str], image_size: List[int],
    #                         resize: bool, to_cpu: bool):
    #     """ Extract Multiple Layers and concatenate them as hypercolumns

    #     Args:
    #         image: A list of image paths.
    #         image_size: The maximum image size.
    #         resize: Whether images should be resized when loaded.
    #         to_cpu: Whether the resulting hypercolumns should be moved to cpu.
    #     Returns:
    #         hypercolumn: The extracted hypercolumn.
    #         image_resolution: The image resolution used as input.
    #     """
    #     # Pass list of image paths and compute descriptors
    #     with torch.no_grad():

    #         # Extract tensor from image
    #         feature_map = ImagesFromList.image_path_to_tensor(
    #             image_paths=image,
    #             image_size=image_size,
    #             resize=resize,
    #             device=self._device)
    #         image_resolution = feature_map[0].shape[1:]
    #         feature_maps, j = [], 0
    #         for i, layer in enumerate(list(self._model.encoder.children())):
    #             if(j==len(self._hypercolumn_layers)):
    #                 break
    #             if(i==self._hypercolumn_layers[j]):
    #                 feature_maps.append(feature_map)
    #                 j+=1
    #             feature_map = layer(feature_map)

    #         # Final descriptor size (concat. intermediate features)
    #         final_descriptor_size = sum([x.shape[1] for x in feature_maps])
    #         b, c, w, h = feature_maps[0].shape
    #         hypercolumn = torch.zeros(
    #             b, final_descriptor_size, w, h).to(self._device)

    #         # Upsample to the largest feature map size
    #         start_index = 0
    #         for j in range(len(self._hypercolumn_layers)):
    #             descriptor_size = feature_maps[j].shape[1]
    #             upsampled_map = interpolate(
    #                 feature_maps[j], size=(w, h),
    #                 mode='bilinear', align_corners=True)
    #             hypercolumn[:, start_index:start_index + descriptor_size, :, :] = upsampled_map
    #             start_index += descriptor_size

    #         # Delete and empty cache
    #         del feature_maps, feature_map, upsampled_map
    #         torch.cuda.empty_cache()

    #     # Normalize descriptors
    #     hypercolumn = hypercolumn / torch.norm(
    #         hypercolumn, p=2, dim=1, keepdim=True)
    #     # torch.save(hypercolumn,"vgg.pt")
    #     if to_cpu:
    #         hypercolumn = hypercolumn.cpu().data.numpy()
    #     return hypercolumn, image_resolution

    # @property
    # def device(self):
    #     return self._device
