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

    def get_pairs(self, embedding1, embedding2, match_pos):
        # if self.cpu:
        #     embeddings = embeddings.cpu()
       
# TODO: adjust the weight
        scale_position = 1e-2
        weight = 10 
        match_pos_in_1 = match_pos['a'] #//4
        match_pos_in_2 = match_pos['b'] #//4
        B,N,_ = match_pos_in_1.shape
        C = embedding1.shape[1]        

        embedding1 = embedding1.reshape((B,N,C))  # checked
        embedding2 = embedding2.reshape((B,N,C))  # checked

        # compute feature distance
        feature_distance_matrix = get_pdist(embedding1, embedding2) # [row i,:] row i of embedding1 vs. each row of embedding2
        # compute pixel position distance
        position_distance_matrix_in_1 = get_pdist(match_pos_in_1, match_pos_in_1)

        position_distance_matrix_in_2 = get_pdist(match_pos_in_2, match_pos_in_2)
        # add identity to set the diagnal a large value so that no positive matches are selected
        y = batched_eye_like(feature_distance_matrix, feature_distance_matrix.shape[1])
        eps = 10000

        t1 = weight*torch.exp(-position_distance_matrix_in_1*scale_position)
        t2 = weight*torch.exp(-position_distance_matrix_in_2*scale_position)
        # max_ = torch.max(t1)
        # min_ = torch.min(t1)
        # tmp1 = torch.mean(feature_distance_matrix)
        # tmp2 = torch.mean(t1)

        weighted_distance_matrix_1_to_2 = feature_distance_matrix + t2 + y*eps
        weighted_distance_matrix_2_to_1 = feature_distance_matrix.transpose(2,1) + t1 + y*eps

        match_neg_in_1 = torch.zeros(match_pos_in_1.shape)
        match_neg_in_2 = torch.zeros(match_pos_in_1.shape)
        # min_, neg_idx = torch.min(weighted_distance_matrix_1_to_2, dim = -1, keepdim=True) # checked

        for batch_i in range(match_pos_in_1.shape[0]):
            # m = match_pos[batch_i]
            # n = neg_idx[batch_i].view(-1)
            weight_1_2 = weighted_distance_matrix_1_to_2[batch_i][:N//2,:] # half of the negative matches are from 1 to 2, first half of 1 to 2
            weight_2_1 = weighted_distance_matrix_2_to_1[batch_i][N//2:,:] # another half of the negative matches are from 2 to 1, second half of 2 to 1
            
            _, neg_idx_2 = torch.min(weight_1_2, dim = -1, keepdim=True)
            match_neg_in_2_part1 = torch.index_select(match_pos_in_2[batch_i], 0, neg_idx_2.view(-1)) # checked
            match_neg_in_2[batch_i] = torch.cat((match_neg_in_2_part1, match_pos_in_2[batch_i, N//2:,:]),dim = 0)

            _, neg_idx_1 = torch.min(weight_2_1, dim = -1, keepdim=True)
            match_neg_in_1_part2 = torch.index_select(match_pos_in_1[batch_i], 0, neg_idx_1.view(-1)) # checked
            match_neg_in_1[batch_i] = torch.cat((match_pos_in_1[batch_i,:N//2,:],match_neg_in_1_part2), dim = 0)

        match_neg = {'a':match_neg_in_1, 'b':match_neg_in_2}
        return match_neg

# def pdist(vectors):
#     distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
#         dim=1).view(-1, 1)
#     return distance_matrix

def get_pdist(data1, data2):
    # data1, data2: BxNx2
    diff = data1[:, :, None, :] - data2[:,None,:,:]
    pdist = torch.sum(diff*diff, axis = -1)
    return torch.sqrt(pdist)

def batched_eye_like (x, n):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)
    

