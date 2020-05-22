import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import wandb
from enum import Enum
from utils import bilinear_interpolation, batched_eye_like, torch_gradient, MyFunctionNegativeTripletSelector, extract_features, normalize_
from corres_sampler import random_select_positive_matches


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


class GNLoss(nn.Module):
    '''
    GN loss function.
    '''

    def __init__(self, margin=1, contrastive_lamda = 100, gn_lamda=0.3, img_scale=2, e1_lamda = 1, e2_lamda = 2/7, num_matches=1024):
        super(GNLoss, self).__init__()
        self.margin = margin
        # self.pair_selector = MyHardNegativePairSelector()
        self.pair_selector = MyFunctionNegativeTripletSelector(margin=self.margin)
        self.gn_lamda = gn_lamda
        self.contrastive_lamda = contrastive_lamda
        self.img_scale = img_scale  # original colored image is scaled by a factor img_scale.
        self.e1_lamda = e1_lamda
        self.e2_lamda = e2_lamda
        self.num_matches = num_matches

    def compute_contrastive_loss(self, fa, fb, pos):
        N = fa.shape[0]
        diff = fa - fb
        D_feat = torch.norm(diff, p=2, dim=1)
        # length = len(D_feat[D_feat<1])
        if pos:
            return torch.sum(torch.pow(D_feat,2) / N)
        mdist = self.margin - D_feat
        # modified for debugging, 5.16
        # length = len(D_feat[D_feat<self.margin])
        # print(length, " ", torch.mean(D_feat[D_feat<self.margin]).item())
        mdist = torch.clamp(mdist, min=0.0)
        return torch.sum(torch.pow(mdist,2) / N)

    # @torchsnooper.snoop()
    def compute_gn_loss(self, f_t, fb, ub, level, train_or_val):
        '''
        f_t: target features F_a(ua)
        fb: feature map b, BxCxHxW
        ub: pos matches of ua in b
        '''
        # compute start point and its feature
        ub = ub.to(device)
        B, N, _ = ub.shape
        # xs = torch.round(torch.rand(ub.shape) + ub) #round this causes bugs in bilinear interpolation!!! all weights becomes 0!

        # xs = torch.FloatTensor(ub.shape).normal_(0).to(device) + ub.to(device)
        '''can try uniform'''
        xs = torch.FloatTensor(ub.shape).uniform_(-1,1).to(device) + ub
        f_s = extract_features(fb, xs)
        # compute residual
        f_t = normalize_(f_t)
        f_s = normalize_(f_s)
        # f_t = F.normalize(f_t, p=2, dim=-1)
        # f_s = F.normalize(f_s, p=2, dim=-1)
        r = f_s - f_t
        # compute Jacobian

        # modified
        f_s_gx, f_s_gy = torch_gradient(fb)  
        J_xs_x = extract_features(f_s_gx, xs)
        J_xs_y = extract_features(f_s_gy, xs)

        # f_s_gy, f_s_gx = np.gradient(fb.cpu().detach().numpy(), axis=(2, 3)) # numerical derivative: J = d(fb)/d(xs), feature gradients # to check if it is correct
        # J_xs_x = self.extract_features(torch.from_numpy(f_s_gx), xs)#.cuda()
        # J_xs_y = self.extract_features(torch.from_numpy(f_s_gy), xs)#.cuda()

        J = torch.stack([J_xs_x, J_xs_y], dim=-1)  # todo: check dim
        # compute Heissian
        eps = 1e-9  # for invertibility, need to be smaller
        H = (J.transpose(1, 2) @ J + eps * batched_eye_like(J, J.shape[2]))
        # r = r.reshape((r.shape[0], r.shape[1], 1))
        b = J.transpose(1, 2) @ r[..., None]
        # xs_reshape = xs.reshape(B * N, 2, 1)
        miu = xs.reshape(B * N, 2, 1) - torch.inverse(H) @ b
        # first error term
        e1 = 0.5 * ((ub.reshape(B * N, 2, 1) - miu).transpose(1, 2)).type(torch.float32) @ H @ \
            (ub.reshape(B * N, 2, 1) - miu).type(torch.float32)
        e1 = torch.sum(e1)
        # second error term
        log_det = torch.log(torch.det(H)).to(device)
        e2 = B * N * torch.log(torch.tensor(2 * np.pi)).to(device) - 0.5 * log_det.sum(-1).to(device)
        # e = e1 + 2 * e2 / 7
        e = self.e1_lamda * e1 + self.e2_lamda * e2
        # print("e1: ",e1," e2: ", e2)
        return e, e1, e2


    def forward(self, F_a, F_b, positive_matches, epoch, train_or_val):
        '''
        F_a is a list containing 4 feature maps of different shapes
        1: B x C X H/8 x W/8
        2: B x C X H/4 x W/4
        3: B x C X H/2 x W/2
        4: B x C X H x W

        known_matches is the positive matches sampled by dataloader.
        {'a':BxNx2,'b':BxNx2}

        '''
        # for c in known_matches:
        #     c = {key: c[key].to(device) for key in c}
        
        # modified: if use hard
        # positive_matches = known_matches[0]
        # negative_matches = known_matches[1]
        # positive_matches = known_matches
        self.max_size_x = F_a[-1].shape[3]  # B x C x H x W
        self.max_size_y = F_a[-1].shape[2]
        '''get neg pairs, do it only for the finest feature'''
        # slice features for positive matches
        # fa_4_sliced = extract_features(F_a[-1],
                                            # positive_matches['a'] / self.img_scale)  # //4 if the input image is scaled
        # fa_4_sliced_reshape = fa_4_sliced.reshape((B,N,C))
        # fb_4_sliced = extract_features(F_b[-1], positive_matches['b'] / self.img_scale)  # //4
        # fb_4_sliced_reshape = fb_4_sliced.reshape((B,N,C))
        # get hard negative samples
        # negative_matches = self.pair_selector.get_pairs(fa_4_sliced, fb_4_sliced, known_matches, self.img_scale) #//.cuda 4 inside pair selector
        
        '''compute loss for each scale'''
        loss = 0
        tripletloss = 0
        gnloss = 0
        e1 = 0
        e2 = 0

        # added
        tripletloss_level = []
        gnloss_level = []
        loss_pos_mean_level = []
        loss_neg_mean_level = []
        num_matches_list = [1000, 2000, 4000, 8000]

        N = positive_matches['a'].shape[1]  # the number of pos and neg matches
        for i in range(len(F_a)):
            level = np.power(2, 3 - i)
            # if (i == 3):
            #     fa_sliced_pos = fa_4_sliced
            #     fb_sliced_pos = fb_4_sliced
            # else:
            positive_matches_sampled = random_select_positive_matches(positive_matches['a'], positive_matches['b'], num_of_pairs=num_matches_list[i])
            fa_sliced_pos = extract_features(F_a[i], positive_matches_sampled['a'] / (level * self.img_scale)) #(level*4)) #TODO: use bilinear interpolation in extract_features
            # fa_sliced_pos = extract_features(F_a[i], positive_matches['a'] / (level * self.img_scale)) #(level*4)) #TODO: use bilinear interpolation in extract_features
            # fb_sliced_pos = extract_features(F_b[i], positive_matches['b'] / (level * self.img_scale)) #(level*4))
            # fa_sliced_neg = self.extract_features(F_a[i], negative_matches['a'] / level) # don't //4 here. negative_matches are in the same scale as known_matches//4
            # fb_sliced_neg = self.extract_features(F_b[i], negative_matches['b'] / level)
            # fa_sliced_neg = extract_features(F_a[i], negative_matches['a'] / level) #(level*self.img_scale)) img_scale is divided inside pair selector
            # fb_sliced_neg = extract_features(F_b[i], negative_matches['b'] / level) #(level*self.img_scale))
            
            '''compute contrastive loss'''
            # loss of positive pairs:
            # loss_pos = self.compute_contrastive_loss(fa_sliced_pos, fb_sliced_pos,pos=True)
            # loss_pos = 0 # modified for debugging, 5.16
            # loss of negative pairs:
            # loss_neg = self.compute_contrastive_loss(fa_sliced_neg, fb_sliced_neg, pos=False)

            '''compute triplet loss'''
            # loss_triplet = self.compute_triplet_loss(fa_sliced_pos, fb_sliced_pos, fb_sliced_neg)
            topM = np.clip(64*np.exp(-epoch*0.6/1000), a_min = 5, a_max=None)
            loss_triplet, loss_pos_mean, loss_neg_mean = self.pair_selector.get_triplets(F_a[i], F_b[i], positive_matches_sampled, self.img_scale*level, topM = int(topM), dist_threshold=0.2, train_or_val=train_or_val)
            # to keep very level the same loss scale
            # loss_triplet /= (i+1)
            
            tripletloss_level.append(loss_triplet)
            loss_pos_mean_level.append(loss_pos_mean)
            loss_neg_mean_level.append(loss_neg_mean)

            '''compute gn loss'''
            loss_gn_all = self.compute_gn_loss(fa_sliced_pos, F_b[i], positive_matches_sampled['b'] / (level * self.img_scale), level, train_or_val)  # //4
            loss_gn = loss_gn_all[0]
            gnloss_level.append(loss_gn)
            # loss = self.contrastive_lamda*(loss_pos + loss_neg) + (self.gn_lamda * loss_gn) + loss
            loss = self.contrastive_lamda*loss_triplet + (self.gn_lamda * loss_gn) + loss 

            # contras_loss = self.contrastive_lamda*(loss_pos + loss_neg) + contras_loss
            gnloss = (self.gn_lamda * loss_gn) + gnloss
        
            tripletloss = (self.contrastive_lamda * loss_triplet) + tripletloss
            e1 = e1 + loss_gn_all[1]
            e2 = e2 + loss_gn_all[2]

            # pos_loss = self.contrastive_lamda*loss_pos + pos_loss
            # neg_loss = self.contrastive_lamda*loss_neg + neg_loss
        # print('contrastive loss: {}, gn loss: {}'.format((loss_pos + loss_neg), self.lamda * loss_gn))
        return loss, tripletloss, gnloss, tripletloss_level, gnloss_level, e1, e2, loss_pos_mean_level, loss_neg_mean_level