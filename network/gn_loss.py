import torch
import torch.nn as nn
import numpy as np
# import torchsnooper
from utils import MyHardNegativePairSelector, bilinear_interpolation, batched_eye_like, torch_gradient

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


# TODO: resize the image and corresponding matches

class GNLoss(nn.Module):
    '''
    GN loss function.
    '''

    def __init__(self, margin=1.0, lamda=0.3, img_scale=2):
        super(GNLoss, self).__init__()
        self.margin = margin
        self.pair_selector = MyHardNegativePairSelector()
        self.lamda = lamda
        self.img_scale = img_scale  # original colored image is scaled by a factor img_scale.

    def extract_features(self, f, indices):
        '''
        f: BxCxHxW
        indicies: BxNx2
        '''
        B, C, H, W = f.shape
        N = indices.shape[1]
        for b in range(f.shape[0]):
            f_bth = bilinear_interpolation(f[b, :, :, :], indices[b, :, :])
            if not b:
                f_2d = f_bth
            else:
                f_2d = torch.cat((f_2d, f_bth), dim=1)
        return f_2d.transpose(0, 1)

        # N = indices.shape[1]
        # b, c, h, w = f.shape
        # f_permuted = f.permute(1,0,2,3)
        # f_2d = f_permuted.reshape((c, b*h*w))
        # f_idx_2d = np.zeros((b*N)) # for cpu debugging
        # # modified
        # # f_idx_2d_cuda_tensor = torch.zeros((b*N)).cuda()
        # for b_th in range(b):
        #     m = indices[b_th]
        #     f_idx_2d[b_th*(N):(b_th+1)*N] = b_th*w*h + m[:,1]*w + m[:,0]
        #     # modified
        #     # f_idx_2d_cuda_tensor[b_th*(N):(b_th+1)*N] = b_th*w*h + m[:,1]*w + m[:,0]
        #     # f_idx_2d = f_idx_2d_cuda_tensor.cpu().numpy()
        # f_idx_2d = torch.floor(torch.from_numpy(f_idx_2d)).type(torch.LongTensor).to(f_2d.device)

        # return torch.index_select(f_2d, -1, f_idx_2d).transpose(0,1)
        # return torch.index_select(f_2d, -1, f_idx_2d)

    def compute_contrastive_loss(self, fa, fb, N, pos):

        diff = fa - fb
        if pos:
            return torch.sum(torch.pow(diff, 2) / N)
        mdist = self.margin - diff
        mdist = torch.clamp(mdist, min=0.0)

        return torch.sum(torch.pow(mdist, 2) / N)

    # @torchsnooper.snoop()
    def compute_gn_loss(self, f_t, fb, ub):
        '''
        f_t: target features F_a(ua)
        fb: feature map b, BxCxHxW
        ub: pos matches of ua in b
        '''
        # compute start point and its feature
        # ub = corres_pos['b']
        B, N, _ = ub.shape
        # xs = torch.round(torch.rand(ub.shape) + ub) #round this causes bugs in bilinear interpolation!!! all weights becomes 0!

        # p_x = torch.clamp(xs[:,:,0], max = (self.max_size_x-1) // level, min = 0)
        # p_y = torch.clamp(xs[:,:,1], max = (self.max_size_y-1) // level, min = 0)
        # xs = torch.stack((p_x, p_y), dim = -1)
        # tmp,_ = torch.max(xs[:,:,0],dim=-1)
        # tmp,_ = torch.max(xs[:,:,1], dim = -1)
        # xs = torch.round(torch.rand(ub.shape) + ub)
        # check if go beyound boundaries

        # xs = torch.round(torch.rand(ub.shape) + ub) # start at most 1 pixel away from u_b
        xs = 3 * torch.FloatTensor(ub.shape).uniform_(-1, 1).to(device) + ub.to(device)
        # xs = torch.rand(ub.shape).to(device) + ub.to(device)
        # torch.clamp(min=0, max = self.max_size[1], xs[:]) # self.max_size: H x W
        f_s = self.extract_features(fb, xs)
        # compute residual
        r = f_s - f_t
        # compute Jacobian

        # modified
        f_s_gx, f_s_gy = torch_gradient(fb)  # problematic
        J_xs_x = self.extract_features(f_s_gx, xs)
        J_xs_y = self.extract_features(f_s_gy, xs)

        # f_s_gy, f_s_gx = np.gradient(fb.cpu().detach().numpy(), axis=(2, 3)) # numerical derivative: J = d(fb)/d(xs), feature gradients # to check if it is correct
        # J_xs_x = self.extract_features(torch.from_numpy(f_s_gx), xs)#.cuda()
        # J_xs_y = self.extract_features(torch.from_numpy(f_s_gy), xs)#.cuda()

        J = torch.stack([J_xs_x, J_xs_y], dim=-1).type(torch.float32)  # todo: check dim
        # compute Heissian
        eps = 1e-9  # for invertibility, need to be smaller
        H = (J.transpose(1, 2) @ J + eps * batched_eye_like(J, J.shape[2]))
        r = r.reshape((r.shape[0], r.shape[1], 1)).type(torch.float32)
        b = J.transpose(1, 2) @ r
        xs_reshape = xs.reshape(B * N, 2, 1)
        miu = xs_reshape - torch.inverse(H) @ b.type(torch.float32)
        # first error term
        e1 = 0.5 * ((ub.reshape(B * N, 2, 1) - miu).transpose(1, 2)).type(torch.float32) @ H @ (
                    ub.reshape(B * N, 2, 1) - miu).type(torch.float32)  # check dim, very unsure
        e1 = torch.sum(e1)
        # second error term
        log_det = torch.log(torch.det(H)).to(device)
        e2 = B * N * torch.log(torch.tensor(2 * np.pi)).to(device) - 0.5 * log_det.sum(-1).to(device)
        e = e1 + 2 * e2 / 7
        return e

    def forward(self, F_a, F_b, known_matches):
        '''
        F_a is a list containing 4 feature maps of different shapes
        1: B x C X H/8 x W/8
        2: B x C X H/4 x W/4
        3: B x C X H/2 x W/2
        4: B x C X H x W

        known_matches is the positive matches sampled by dataloader.
        {'a':BxNx2,'b':BxNx2}

        TODO: use for loop to check pair selector MyHardNegativePairSelector
        '''
        self.max_size_x = F_a[-1].shape[3]  # B x C x H x W
        self.max_size_y = F_a[-1].shape[2]
        '''get neg pairs, do it only for the finest feature'''
        # slice features for positive matches
        fa_4_sliced = self.extract_features(F_a[-1],
                                            known_matches['a'] / self.img_scale)  # //4 if the input image is scaled
        # fa_4_sliced_reshape = fa_4_sliced.reshape((B,N,C))
        fb_4_sliced = self.extract_features(F_b[-1], known_matches['b'] / self.img_scale)  # //4
        # fb_4_sliced_reshape = fb_4_sliced.reshape((B,N,C))
        # get hard negative samples
        negative_matches = self.pair_selector.get_pairs(fa_4_sliced, fb_4_sliced, known_matches,
                                                        self.img_scale)  # //.cuda 4 inside pair selector

        '''compute loss for each scale'''
        loss = 0
        N = known_matches['a'].shape[1]  # the number of pos and neg matches
        for i in range(len(F_a)):
            level = np.power(2, 3 - i)
            if (i == 3):
                fa_sliced_pos = fa_4_sliced
                fb_sliced_pos = fb_4_sliced
            else:
                fa_sliced_pos = self.extract_features(F_a[i], known_matches['a'] / (
                            level * self.img_scale))  # (level*4)) #TODO: use bilinear interpolation in extract_features
                fb_sliced_pos = self.extract_features(F_b[i],
                                                      known_matches['b'] / (level * self.img_scale))  # (level*4))
            fa_sliced_neg = self.extract_features(F_a[i], negative_matches[
                'a'] / level)  # don't //4 here. negative_matches are in the same scale as known_matches//4
            fb_sliced_neg = self.extract_features(F_b[i], negative_matches['b'] / level)

            '''compute contrastive loss'''
            # loss of positive pairs:
            loss_pos = self.compute_contrastive_loss(fa_sliced_pos, fb_sliced_pos, N, pos=True)
            # loss of negative pairs:
            loss_neg = self.compute_contrastive_loss(fa_sliced_neg, fb_sliced_neg, N, pos=False)

            '''compute gn loss'''
            loss_gn = self.compute_gn_loss(fa_sliced_pos, F_b[i], known_matches['b'] / (level * self.img_scale))  # //4
            loss = (loss_pos + loss_neg) + (self.lamda * loss_gn) + loss
        print('contrastive loss: {}, gn loss: {}'.format((loss_pos + loss_neg), self.lamda * loss_gn))
        return loss