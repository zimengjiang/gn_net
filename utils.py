from itertools import combinations
from enum import Enum
import shutil, os
import numpy as np
import torch
import torch.nn.functional as F
# import wandb
# import torchsnooper
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

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
    dist[dist != dist] = 1e-16  # replace nan values with 0
    return torch.clamp(dist, 1e-16, np.inf)


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

    return weight00 * grid00 + weight01 * grid01 + weight10 * grid10 + weight11 * grid11


def torch_gradient(f):
    """sobel filter"""
    # f: BxCxHxW
    b, c, h, w = f.shape
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

def np_gradient_filter(f):
    # f: BxCxHxW
    b, c, h, w = f.shape
    np_gradient_y = torch.FloatTensor([[0., -0.5, 0.], [0., 0., 0.], [0., 0.5, 0.]]).view(1, 1, 3, 3).to(f)
    np_gradient_x = torch.FloatTensor([[0., 0., 0], [-0.5, 0., 0.5], [0., 0., 0]]).view(1, 1, 3, 3).to(f)
    f_gradx = F.conv2d(f.view(-1, 1, h, w), np_gradient_x, stride=1, padding=1).view(b, c, h, w)
    f_grady = F.conv2d(f.view(-1, 1, h, w), np_gradient_y, stride=1, padding=1).view(b, c, h, w)
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



class MyFunctionNegativeTripletSelector():
    """
    Given positive pairs, sample topM hardest negatives and return double margin contrastive loss. 
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

        a1 = match_pos['a'] / scale  # positive matches in img1
        a2 = match_pos['b'] / scale  # positive matches in img2
        B, C, H, W = embedding1.shape
        N = a1.shape[1]
        # slice positive features
        e1_sliced_ = extract_features(embedding1, a1)  #(BxN)*C
        e1_sliced = e1_sliced_.reshape((B, N, C))  #BxNxC
        e2_sliced_ = extract_features(embedding2, a2) #(BxN)*C


        e2 = embedding2.reshape((B, C, -1))  
        e2 = e2.transpose(1, 2)   

        # if normalized L2 norm then euclidean distance，but we found unnormalized distance is better.
        # e1_sliced = F.normalize(e1_sliced, p=2, dim=-1)
        # e2 = F.normalize(e2, p = 2, dim=-1)
        # e2_sliced_ = F.normalize(e2_sliced_, p=2, dim=-1)
        # e1_sliced_ = F.normalize(e1_sliced_, p=2, dim=-1)
        f_dist_a1_img2 = batch_pairwise_squared_distances(e1_sliced,e2) # dim: B x #a1 x #pixels in img2

        # get all pixel positions of img2
        idx_1d = torch.arange(H * W)
        idx_x = idx_1d % W
        idx_y = idx_1d // W
        idx_xy = torch.stack((idx_x, idx_y), dim=1)
        idx_batched_xy = idx_xy.repeat(B, 1, 1)
        # apply distance constrain. distance smaller than threshold will cause very large loss and won't be sampled 
        p_dist_12 = batch_pairwise_squared_distances(a2, idx_batched_xy)
        mask_12 = p_dist_12 < (dist_threshold*H)**2
        f_dist_a1_img2[mask_12] = 1e4
        # for each keypoint in img1, compute topM hardest negative matches in img2.
        dist_nn12, idx_in_2 = f_dist_a1_img2.topk(topM, dim=-1, largest=False)
        dist_nn12 = dist_nn12.reshape(B * N, -1)
        idx_in_2 = idx_in_2.reshape(B * N, -1)
        # randomly sample among topM hardest negative matches 
        D_feat_neg = torch.clamp(torch.sqrt(dist_nn12[torch.arange(B * N),sampled_neg_idx]), min=1e-16) # avoid invalid operation when taking derivative w.r.t sqrt.
        # compute negative loss
        loss_neg = torch.clamp(self.margin_neg - D_feat_neg, min=0.0)
        loss_neg = loss_neg**2
        # compute positive loss
        D_feat_pos = torch.clamp(torch.norm(e1_sliced_ - e2_sliced_, p=2, dim=1), min = 1e-16)
        loss_pos = torch.clamp(D_feat_pos - self.margin_pos, min = 0.0)
        loss_pos = loss_pos**2

        mdist = loss_neg + loss_pos
        # compute mean loss
        loss_pos_mean = torch.mean(loss_pos, dim=-1)
        loss_neg_mean = torch.mean(loss_neg, dim=-1)
        return torch.sum(mdist), loss_pos_mean, loss_neg_mean