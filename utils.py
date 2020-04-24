from itertools import combinations

import numpy as np
import torch

class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embedding1, embedding2):
        raise NotImplementedError

class MyHardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(MyHardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embedding1, embedding2, match_pos_in_1):
        # if self.cpu:
        #     embeddings = embeddings.cpu()

        B,N,_ = match_pos_in_1.shape
        C = embedding1.shape[1]
        embedding1 = embedding1.reshape((B,N,C))
        embedding2 = embedding2.reshape((B,N,C))

        distance_matrix = get_pdist(embedding1, embedding2)
        # add identity to set the diagnal a large value so that no positive matches are selected
        x = torch.eye(distance_matrix.shape[1])
        x = x.reshape((1, distance_matrix.shape[1], distance_matrix.shape[1]))
        y = x.repeat(distance_matrix.shape[0], 1, 1)
        # modified
        y = y.cuda()
        eps = 10000
        distance_matrix = distance_matrix + y*eps
        match_neg_in_2 = torch.zeros(match_pos_in_1.shape)
        min_, neg_idx = torch.min(distance_matrix, dim = -1, keepdim=True)
        for batch_i in range(match_pos_in_1.shape[0]):
            # m = match_pos[batch_i]
            # n = neg_idx[batch_i].view(-1)
            match_neg_in_2[batch_i] = torch.index_select(match_pos_in_1[batch_i], 0, neg_idx[batch_i].view(-1))
        
        match_neg = {'a':match_pos_in_1, 'b':match_neg_in_2}
        return match_neg

# def pdist(vectors):
#     distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
#         dim=1).view(-1, 1)
#     return distance_matrix

def get_pdist(data1, data2):
    # data1, data2: BxNx2
    diff = data1[:, :, None, :] - data2[:,None,:,:]
    pdist = torch.sum(diff*diff, axis = -1)
    return pdist
    

