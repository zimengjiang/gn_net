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


def getitem(data, idx):
    a = data['corres_pos_all']['a'][idx].squeeze()
    b = data['corres_pos_all']['b'][idx].squeeze()
    pos_a, pos_b = random_select_positive_matches(a, b, num_of_pairs=1024)
    # neg_a, neg_b = random_select_negative_matches(a, b, num_of_pairs=1024)
    corres_ab_pos = {'a':pos_a, 'b':pos_b}
    # corres_ab_neg = {'a':neg_a, 'b':neg_b}
    return corres_ab_pos


def count():
    data = {
            'name': 'cmu',
            'root': '/cluster/work/riner/users/PLR-2020/lechen/gn_net/gn_net_data',
            'slice_folder': None,
            'image_folder': 'images',
            'pair_info_folder': 'correspondence',
            'pair_file_names': None,
            'queries_folder': 'query',
            'image_pairs_name': None, 
            'corres_pos_all': None,
            'scale': 4
            }

    # root= '/local/home/lixxue/gnnet/gn_net_data'
    # name= 'cmu'
    # pair_info_folder= 'correspondence'
    # image_folder= 'images'
    cmu_slice_all = True

    # load image pairs for one slice
    # /public_data/cmu/corrrespondence
    pair_file_roots = Path(data['root'], data['name'] ,data['pair_info_folder'])
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
    data['pair_file_names'] = pair_files
    # print(pair_files)

    filename ='count.txt'
    with open(filename,'a') as file_object:
        file_object.write('total:'+' '+str(len(pair_files))+'\n\n')

    N = len(data['pair_file_names']) # number of image pairs
    image_pairs = {'a':[], 'b':[]}
    corres_all_pos = {'a':[], 'b':[]}

    num = 0

    # /public_data/cmu/images
    # query_root = Path(self._data['root'], self._data['name'], self._data['image_folder'], self._data['slice_folder'], self._data['queries_folder'])
    
    for f in data['pair_file_names']:
        # pair_info = scipy.io.loadmat(f)
        pair_info = h5py.File(f)
        print(pair_info)
        org_name_a = u''.join(chr(c) for c in pair_info['im_i_path'])
        org_name_b = u''.join(chr(c) for c in pair_info['im_j_path'])
        query_root = Path(data['root'], data['name'], data['image_folder'], (f.split('/')[-1]).split('_')[1], data['queries_folder'])
        image_pairs['a'].append(Path(query_root, org_name_a.split('/')[-1]))
        image_pairs['b'].append(Path(query_root, org_name_b.split('/')[-1]))
        # corres_all_pos['a'].append(pair_info['pt_i'].value)
        corres_all_pos['a'] = pair_info['pt_i'][()]
        corres_all_pos['b'] = pair_info['pt_j'][()]  # N x 2
        a = corres_all_pos['a'].squeeze()
        b = corres_all_pos['b'].squeeze()

        # print(len(a))
        # print(len(b))
        if len(a)<500:
            num = num+1
            filename ='count.txt'
            with open(filename,'a') as file_object:
                file_object.write('a:'+' '+ org_name_a + '\n' + 'b:'+' '+ org_name_b + '\n' + 'num:'+str(len(a))+'\n\n')
    print(num)
    filename ='count.txt'
    with open(filename,'a') as file_object:
        file_object.write('count:'+' '+str(num)+'\n\n')

    # data['image_pairs_name'] = image_pairs
    # data['corres_pos_all'] = corres_all_pos


if __name__ == '__main__':
    count()