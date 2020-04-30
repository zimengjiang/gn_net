import numpy as np
from PIL import Image
from pathlib import Path
from glob import glob 
from corres_sampler import random_select_positive_matches
import scipy.io
import h5py # for loading v7.3 .mat 

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.transforms import transforms

"""
Train: For each image pair creates randomly positive and negative matches
    cross-season-correspondence dataset, CMU
    todo: check if all image pairs are in query list  

Initialize CMU class attributes.
        Args:
            root: The root to the dataset folder.
            image_folder: The folder containing the images.
            pair_info_folder: The folder containing .mat file 
                        that stores image pairs and their positive correspondences.
            name: The dataset name.
            database_folder: The subfolder name containing the database images. Not used here. 
            queries_folder: The subfolder name containing the query images.
            cmu_slice: The index of the CMU slice.
            image_pairs_name: The dict storing the path(name) to image pairs.
            corres_pos_all: The dict storing all the positive correspondences given by .mat files. 
"""

class CMUDataset(Dataset): 
    def __init__(self,  root:str,
                        image_folder:str,
                        pair_info_folder: str,
                        name: str = None,
                        queries_folder: str = None,
                        cmu_slice_all: bool = True,
                        cmu_slice: int =  None,
                        transform = None,
                        img_scale: int = None
                        ):
        self._data = {
            'name': 'cmu',
            'root': root,
            'slice_folder': None,
            'image_folder': image_folder,
            'pair_info_folder': pair_info_folder,
            'pair_file_names': None,
            'queries_folder': queries_folder,
            'image_pairs_name': None, 
            'corres_pos_all': None,
            'scale': img_scale
        }
        if not cmu_slice_all:
            self._data['slice_folder'] = 'slice{}'.format(cmu_slice)
        else:
            self._data['slice_folder'] = ['slice{}'.format(s) for s in range(2,26)]
        self.load_pair_file_names(cmu_slice, cmu_slice_all)
        self.load_image_pairs(cmu_slice, cmu_slice_all)
        self.transform = transform
        self.default_transform = self.default_transform()
    
    def load_pair_file_names(self, cmu_slice, cmu_slice_all):
        # load image pairs for one slice
        # /public_data/cmu/corrrespondence
        pair_file_roots = Path(self._data['root'], self._data['name'] ,self._data['pair_info_folder'])
        # /public_data/cmu/corrrespondence/*.mat
        if not cmu_slice_all:
            suffix  = 'correspondence_slice{}*.mat'.format(cmu_slice)
        else:
            suffix = '*.mat'
        pair_files = glob(str(Path(pair_file_roots, suffix)))
        if not len(pair_files):
            raise Exception('No correspondence file found at {}'.format(pair_file_roots))
        if not cmu_slice_all:
            print (('>> Found {} image pairs for slice {}').format(len(pair_files), cmu_slice))
        else:
            print (('>> Found {} image pairs for all slice').format(len(pair_files)))
        self._data['pair_file_names'] = pair_files
    
    def load_image_pairs(self, cmu_slice, cmu_slice_all):
        N = len(self._data['pair_file_names']) # number of image pairs
        image_pairs = {'a':[], 'b':[]}
        corres_all_pos = {'a':[], 'b':[]}
        # /public_data/cmu/images
        # query_root = Path(self._data['root'], self._data['name'], self._data['image_folder'], self._data['slice_folder'], self._data['queries_folder'])
        for f in self._data['pair_file_names']:
            # pair_info = scipy.io.loadmat(f)
            pair_info = h5py.File(f)
            org_name_a = u''.join(chr(c) for c in pair_info['im_i_path'])
            org_name_b = u''.join(chr(c) for c in pair_info['im_j_path'])
            query_root = Path(self._data['root'], self._data['name'], self._data['image_folder'], (f.split('/')[-1]).split('_')[1], self._data['queries_folder'])
            image_pairs['a'].append(Path(query_root, org_name_a.split('/')[-1]))
            image_pairs['b'].append(Path(query_root, org_name_b.split('/')[-1]))
            corres_all_pos['a'].append(pair_info['pt_i'][()])
            corres_all_pos['b'].append(pair_info['pt_j'][()])  # N x 2
        self._data['image_pairs_name'] = image_pairs
        self._data['corres_pos_all'] = corres_all_pos

    # To do: double check image W, H !!!!!!
    '''?
    mean and std for normalization? 
    '''
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((768//self._data['scale'], 1024//self._data['scale'])), # check image dim, resize (H, W)?  just for fast debugging 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    '''
    '''
    def __getitem__(self, idx):
        img_a = self._data['image_pairs_name']['a'][idx]
        img_b = self._data['image_pairs_name']['b'][idx]
        a = self._data['corres_pos_all']['a'][idx].squeeze()
        b = self._data['corres_pos_all']['b'][idx].squeeze()
        pos_a, pos_b = random_select_positive_matches(a, b, num_of_pairs=500)
        # neg_a, neg_b = random_select_negative_matches(a, b, num_of_pairs=1024)
        corres_ab_pos = {'a':pos_a, 'b':pos_b}
        # corres_ab_neg = {'a':neg_a, 'b':neg_b}
        if self.transform:
            img_a = self.default_transform(Image.open(img_a))
            img_b = self.default_transform(Image.open(img_b))
        # return (img_a, img_b), (corres_ab_pos,corres_ab_neg)
        return (img_a, img_b), corres_ab_pos

    def __len__(self):
            assert len(self._data['image_pairs_name']['a']) == len (self._data['image_pairs_name']['b'])
            return len(self._data['image_pairs_name']['a'])
        










        
    