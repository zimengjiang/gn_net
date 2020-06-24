from itertools import combinations
from enum import Enum
import shutil, os
import numpy as np
import torch
import torch.nn.functional as F
import wandb
# import torchsnooper
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


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

    def get_pairs(self, embedding1, embedding2, match_pos, img_scale):
        # if self.cpu:
        #     embeddings = embeddings.cpu()

        # TODO: adjust the weight
        scale_position = 1e-2
        weight = 10
        match_pos_in_1 = match_pos['a'] / img_scale  # //4
        match_pos_in_2 = match_pos['b'] / img_scale  # //4
        B, N, _ = match_pos_in_1.shape
        C = embedding1.shape[1]

        embedding1 = embedding1.reshape((B, N, C))  # checked
        embedding2 = embedding2.reshape((B, N, C))  # checked

        # compute feature distance
        feature_distance_matrix = get_pdist(
            embedding1, embedding2
        )  # [row i,:] row i of embedding1 vs. each row of embedding2
        # compute pixel position distance
        position_distance_matrix_in_1 = get_pdist(match_pos_in_1,
                                                  match_pos_in_1)

        position_distance_matrix_in_2 = get_pdist(match_pos_in_2,
                                                  match_pos_in_2)
        # add identity to set the diagnal a large value so that no positive matches are selected
        y = batched_eye_like(feature_distance_matrix,
                             feature_distance_matrix.shape[1])
        eps = 10000

        t1 = weight * torch.exp(
            -position_distance_matrix_in_1 * scale_position)
        t2 = weight * torch.exp(
            -position_distance_matrix_in_2 * scale_position)
        # max_ = torch.max(t1)
        # min_ = torch.min(t1)
        # tmp1 = torch.mean(feature_distance_matrix)
        # tmp2 = torch.mean(t1)

        weighted_distance_matrix_1_to_2 = feature_distance_matrix + t2 + y * eps
        weighted_distance_matrix_2_to_1 = feature_distance_matrix.transpose(
            2, 1) + t1 + y * eps

        match_neg_in_1 = torch.zeros(match_pos_in_1.shape)
        match_neg_in_2 = torch.zeros(match_pos_in_1.shape)
        # min_, neg_idx = torch.min(weighted_distance_matrix_1_to_2, dim = -1, keepdim=True) # checked

        for batch_i in range(match_pos_in_1.shape[0]):
            # m = match_pos[batch_i]
            # n = neg_idx[batch_i].view(-1)
            weight_1_2 = weighted_distance_matrix_1_to_2[
                batch_i][:N //
                         2, :]  # half of the negative matches are from 1 to 2, first half of 1 to 2
            weight_2_1 = weighted_distance_matrix_2_to_1[batch_i][
                N //
                2:, :]  # another half of the negative matches are from 2 to 1, second half of 2 to 1

            _, neg_idx_2 = torch.min(weight_1_2, dim=-1, keepdim=True)
            match_neg_in_2_part1 = torch.index_select(
                match_pos_in_2[batch_i], 0, neg_idx_2.view(-1))  # checked
            match_neg_in_2[batch_i] = torch.cat(
                (match_neg_in_2_part1, match_pos_in_2[batch_i, N // 2:, :]),
                dim=0)

            _, neg_idx_1 = torch.min(weight_2_1, dim=-1, keepdim=True)
            match_neg_in_1_part2 = torch.index_select(
                match_pos_in_1[batch_i], 0, neg_idx_1.view(-1))  # checked
            match_neg_in_1[batch_i] = torch.cat(
                (match_pos_in_1[batch_i, :N // 2, :], match_neg_in_1_part2),
                dim=0)

        match_neg = {'a': match_neg_in_1, 'b': match_neg_in_2}
        return match_neg


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(
        dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix


def get_pdist(data1, data2, requre_sqrt):
    # data1, data2: BxNx2
    diff = data1[:, :, None, :] - data2[:, None, :, :]
    pdist = torch.sum(diff * diff, axis=-1)
    if requre_sqrt:
        return torch.sqrt(pdist)
    else:
        return pdist


def batch_pairwise_squared_distances(x, y):
    '''                                                                                              
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
    Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
    '''
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)
    x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1], 1)
    y_t = y.permute(0, 2, 1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0], 1, y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0  # replace nan values with 0
    return torch.clamp(dist, 0.0, np.inf)


def normalize_(x):
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    mask = x_norm == 0.0
    x_norm = x_norm + mask * 1e-16
    return x / x_norm

# @torchsnooper.snoop()
def batch_pairwise_cos_distances(x, y, batched):
    '''                                                                                              
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
    Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
    '''
    # x = normalize_(x)
    # y = normalize_(y)
    if batched:
        x = normalize_(x)
        y = normalize_(y)
        y_t = y.permute(0, 2, 1).contiguous()
        cos_simi = torch.bmm(x, y_t)
        return 1 - cos_simi
    else:
        cos_simi = F.cosine_similarity(x, y, dim=-1, eps=1e-16)
        return 1 - cos_simi
        # return 1 - cos_simi
        # return 1 - (x * y).sum(-1)  

def batched_eye_like(x, n):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def extract_features(f, indices):
    '''
    f: BxCxHxW
    indicies: BxNx2
    '''
    # B, C, H, W = f.shape
    # N = indices.shape[1]
    for b in range(f.shape[0]):
        f_bth = bilinear_interpolation(f[b, :, :, :], indices[b, :, :])
        if not b:
            f_2d = f_bth
        else:
            f_2d = torch.cat((f_2d, f_bth), dim=1)
    return f_2d.transpose(0, 1).type(torch.float32)

def extract_features_int(f, indices):
    '''
    f: BxCxHxW
    indicies: BxNx2
    '''
    b, c, h, w = f.shape
    N = indices.shape[1]
    f_permuted = f.permute(1,0,2,3)
    f_2d = f_permuted.reshape((c, b*h*w))
    f_idx_2d = np.zeros((b*N))
    for b_th in range(f.shape[0]):
        m = indices[b_th]
        f_idx_2d[b_th*(N):(b_th+1)*N] = b_th*w*h + m[:,1]*w + m[:,0]
    f_idx_2d = torch.floor(torch.from_numpy(f_idx_2d)).type(torch.LongTensor).to(f_2d.device)
    return torch.index_select(f_2d, -1, f_idx_2d).transpose(0,1)


def bilinear_interpolation(grid, idx):
    # grid: C x H x W
    # idx: N x 2
    _, H, W = grid.shape
    x = idx[..., 0].to(device)
    y = idx[..., 1].to(device)
    x0 = torch.clamp(torch.floor(x), 0, W - 1).to(device)
    y0 = torch.clamp(torch.floor(y), 0, H - 1).to(device)
    x1 = torch.clamp(torch.ceil(x), 0, W - 1).to(device)
    y1 = torch.clamp(torch.ceil(y), 0, H - 1).to(device)
    weight00 = ((x1 - x) * (y1 - y)).to(device)
    weight01 = ((x1 - x) * (y - y0)).to(device)
    weight10 = ((x - x0) * (y1 - y)).to(device)
    weight11 = ((x - x0) * (y - y0)).to(device)
    x0 = x0.type(torch.LongTensor)
    y0 = y0.type(torch.LongTensor)
    x1 = x1.type(torch.LongTensor)
    y1 = y1.type(torch.LongTensor)
    grid00 = grid[..., y0, x0]
    grid01 = grid[..., y0, x1]
    grid10 = grid[..., y1, x0]
    grid11 = grid[..., y1, x1]
    # print(weight00)
    # print(weight01)
    # print(weight10)
    # print(weight11)

    return weight00 * grid00 + weight01 * grid01 + weight10 * grid10 + weight11 * grid11


def torch_gradient(f):
    # f: BxCxHxW
    # sobel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1,1,3,3).expand(1,f.shape[1],3,3) # .cuda()
    # sobel_x = torch.FloatTensor([[-1., 0., 1.],[-2., 0., 2.],[-1., 0., 1.]]).view(1,1,3,3).expand(1,f.shape[1],3,3) # .cuda()
    # f_gradx = F.conv2d(f, sobel_x, stride=1, padding=1)
    # f_grady = F.conv2d(f, sobel_y, stride=1, padding=1)

    b, c, h, w = f.shape
    # x = torch.randn(batch_size, channels, h, w)
    # conv = nn.Conv2d(1, 1, 4, 2, 1)
    # output = conv(x.view(-1, 1, h, w)).view(batch_size, channels, h//2, w//2)
    # sobel operator torch.grad = False checked.
    sobel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.],
                                 [1., 2., 1.]]).view(1, 1, 3,
                                                     3).to(device)  # .cuda()
    sobel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [
        -1., 0., 1.
    ]]).view(1, 1, 3, 3).to(device)  # pytorch performs cross-correlation instead of convolution in information theory
    f_gradx = F.conv2d(f.view(-1, 1, h, w), sobel_x, stride=1,
                       padding=1).view(b, c, h, w)
    f_grady = F.conv2d(f.view(-1, 1, h, w), sobel_y, stride=1,
                       padding=1).view(b, c, h, w)
    return f_gradx, f_grady


def save_checkpoint(model_state,
                    optimizer_state,
                    scheduler_state,
                    is_best,
                    path,
                    epoch,
                    filename='checkpoint.pth.tar'):
    prefix = str(epoch)
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    # torch.save(state, name)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state
            }, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


''' Need Modification!!!'''


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(
        hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(
        np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(
        semihard_negatives) if len(semihard_negatives) > 0 else None


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class MyFunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """
    def __init__(self, margin_pos, margin_neg, margin):
        super(MyFunctionNegativeTripletSelector, self).__init__()
        self.margin = margin
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def get_triplets(self, embedding1, embedding2, match_pos, scale, topM, dist_threshold, train_or_val, level):
        """
        embedding1: feature map of image 1, BxCxHxW
        embedding2: feature map of image 2, BxCxHxW
        match_pos: known positive matches, {'a':BxNx2,'b':BxNx2}
        topM: sort the negatives for each sample by loss in decreasing order and sample randomly over the top M
        dist_threshold: (dist_threshold*H)^2 is the minimal sqaured distance between anchor and neg
        """

        a1 = match_pos['a'] / scale  # anchors in img1
        a2 = match_pos['b'] / scale  # anchors in img2
        B, C, H, W = embedding1.shape
        N = a1.shape[1]
        # slice anchor features
        e1_sliced_ = extract_features(embedding1, a1)  #(BxN)*C
        e1_sliced = e1_sliced_.reshape((B, N, C))  # checked BxNxC
        e2_sliced_ = extract_features(embedding2, a2)


        e2 = embedding2.reshape((B, C, -1))  # checked
        e2 = e2.transpose(1, 2)  # checked 

        # if normalized L2 norm then euclidean distance
        # e1_sliced = F.normalize(e1_sliced, p=2, dim=-1)
        # e2 = F.normalize(e2, p = 2, dim=-1)
        # e2_sliced_ = F.normalize(e2_sliced_, p=2, dim=-1)
        # e1_sliced_ = F.normalize(e1_sliced_, p=2, dim=-1)
        f_dist_a1_img2 = batch_pairwise_squared_distances(e1_sliced,e2) # dim: B x #a1 x #pixels in img2


        # TODO:
        # f_dist_a1_img2 = batch_pairwise_cos_distances(e1_sliced,
                                                    #   e2,
                                                    #   batched=True)
        idx_1d = torch.arange(H * W)
        idx_x = idx_1d % W
        idx_y = idx_1d // W
        idx_xy = torch.stack((idx_x, idx_y), dim=1)
        idx_batched_xy = idx_xy.repeat(B, 1, 1)
        p_dist_12 = batch_pairwise_squared_distances(a2, idx_batched_xy) # modified (a1, idx_batched_xy)
        mask_12 = p_dist_12 < (dist_threshold*H)**2
        f_dist_a1_img2[mask_12] = 1e4

        dist_nn12, idx_in_2 = f_dist_a1_img2.topk(topM, dim=-1, largest=False)
        dist_nn12 = dist_nn12.reshape(B * N, -1)
        idx_in_2 = idx_in_2.reshape(B * N, -1)

        # for visualization
        sampled_neg_idx = torch.randint(0, topM, (B * N, ))
        sampled_neg_idx_in_img2 = idx_in_2[torch.arange(B * N),sampled_neg_idx]
        #todo:
        sampled_neg_idx_x = sampled_neg_idx_in_img2 % W 
        sampled_neg_idx_y = sampled_neg_idx_in_img2 // W
        sampled_neg_idx_xy = torch.stack((sampled_neg_idx_x,sampled_neg_idx_y),dim=1).type(torch.LongTensor)
        e2_sliced_neg = extract_features_int(embedding2, sampled_neg_idx_xy[None, ...])
        e2_sliced_neg_norm_mean = torch.mean(torch.norm(e2_sliced_neg, p=2, dim=-1))
        e1_sliced_pos_norm_mean = torch.mean(torch.norm(e1_sliced_, p=2, dim=-1))
        e2_sliced_pos_norm_mean = torch.mean(torch.norm(e2_sliced_, p=2, dim=-1))
        wandb.log({"feature_norm_a1_level{}".format(level): e1_sliced_pos_norm_mean,
        "feature_norm_a2_level{}".format(level): e2_sliced_pos_norm_mean,
        "feature_norm_n2_level{}".format(level): e2_sliced_neg_norm_mean})

        D_feat_neg = torch.clamp(torch.sqrt(dist_nn12[torch.arange(B * N),sampled_neg_idx]), min=1e-16)
        # loss_neg = torch.clamp(self.margin - D_feat_neg, min=0.0)
        loss_neg = torch.clamp(self.margin_neg - D_feat_neg, min=0.0)
        loss_neg = loss_neg**2
        # loss_neg = 0
        # e1_sliced_ = normalize_(e1_sliced_)
        # e2_sliced_ = normalize_(e2_sliced_)
        # loss_pos = ((e1_sliced_ - e2_sliced_)**2).sum(-1)

        # D_feat_pos = torch.sqrt(((e1_sliced_ - e2_sliced_)**2).sum(-1)) # nan error
        D_feat_pos = torch.clamp(torch.norm(e1_sliced_ - e2_sliced_, p=2, dim=1), min = 1e-16)
        loss_pos = torch.clamp(D_feat_pos - self.margin_pos, min = 0.0)
        loss_pos = loss_pos**2

        # loss_pos = torch.clamp((e1_sliced_ - e2_sliced_)-self.margin_pos, min=0.0)
        # loss_pos = (loss_pos**2).sum(-1)

        # loss_pos = batch_pairwise_cos_distances(e1_sliced_,
        #                                         e2_sliced_,
        #                                         batched=False)
        '''randomly sample a negative'''
        # mdist = torch.clamp(loss_pos - loss_neg + self.margin, min=0.0)
        mdist = loss_neg + loss_pos
        # TODO:
        loss_pos_mean = torch.mean(loss_pos, dim=-1)
        # TODO:
        loss_neg_mean = torch.mean(loss_neg, dim=-1)
        wandb.log({"train_loss_pos_mean": loss_pos_mean, "train_loss_neg_mean": loss_neg_mean})
        '''mean over all negative pairs'''
        # loss_pos = loss_pos.reshape(B*N,1)
        # mdist = torch.clamp(loss_pos - dist_nn12 + self.margin, min=0.0)
        # mdist = torch.mean(mdist, dim=-1)

        # return torch.mean(mdist)
        return torch.sum(mdist), loss_pos_mean, loss_neg_mean
