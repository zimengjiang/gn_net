import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from enum import Enum
from utils import bilinear_interpolation, batched_eye_like, torch_gradient, MyFunctionNegativeTripletSelector, extract_features, normalize_, np_gradient_filter
from corres_sampler import random_select_positive_matches


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


class GNLoss(nn.Module):
    '''
    GN loss function.
    '''

    def __init__(self, margin_pos=0.2, margin_neg=1, margin=1, contrastive_lamda = 100, gn_lamda=0.3, img_scale=2, e1_lamda = 1, e2_lamda = 2/7, num_matches=1024):
        super(GNLoss, self).__init__()
        self.margin = margin
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.pair_selector = MyFunctionNegativeTripletSelector(margin_pos=self.margin_pos, margin_neg=self.margin_neg, margin=self.margin)
        self.gn_lamda = gn_lamda
        self.contrastive_lamda = contrastive_lamda
        self.img_scale = img_scale  # original colored image is scaled by a factor img_scale.
        self.e1_lamda = e1_lamda
        self.e2_lamda = e2_lamda
        self.num_matches = num_matches


    def compute_gn_loss(self, f_t, fb, ub, train_or_val):
        '''
        f_t: target features F_a(ua)
        fb: feature map b, BxCxHxW
        ub: pos matches of ua in b
        '''

        # compute start point and its feature
        ub = ub.to(device)
        B, N, _ = ub.shape
        xs = torch.FloatTensor(ub.shape).uniform_(-1,1).to(device) + ub
        f_s = extract_features(fb, xs)

        # compute residual
        # f_t = normalize_(f_t)
        # f_s = normalize_(f_s)
        r = f_s - f_t

        # compute Jacobian
        f_s_gx, f_s_gy = np_gradient_filter(fb)  
        J_xs_x = extract_features(f_s_gx, xs)
        J_xs_y = extract_features(f_s_gy, xs)
        J = torch.stack([J_xs_x, J_xs_y], dim=-1)  # todo: check dim

        # compute Heissian
        eps = 1e-9  # for invertibility, need to be smaller
        H = (J.transpose(1, 2) @ J + eps * batched_eye_like(J, J.shape[2]))
        b = J.transpose(1, 2) @ r[..., None]
        miu = xs.reshape(B * N, 2, 1) - torch.inverse(H) @ b

        # first error term
        e1 = 0.5 * ((ub.reshape(B * N, 2, 1) - miu).transpose(1, 2)).type(torch.float32) @ H @ \
            (ub.reshape(B * N, 2, 1) - miu).type(torch.float32)
        e1 = torch.sum(e1)

        # second error term
        det_H = torch.clamp(torch.det(H), min=1e-16)
        log_det = torch.log(det_H).to(device)
        e2 = B * N * torch.log(torch.tensor(2 * np.pi)).to(device) - 0.5 * log_det.sum(-1).to(device)
        
        # return gn loss
        e = self.e1_lamda * e1 + self.e2_lamda * e2
        return e, e1, e2


    def forward(self, F_a, F_b, positive_matches, iteration, train_or_val, night_flag):
        '''
        F_a is a list containing 4 feature maps of different shapes
        1: B x C X H/(scale*4) x W/(sclae*4)
        2: B x C X H/(sclae*8) x W/(sclae*8)
        3: B x C X H/(sclae*8) x W/(sclae*8)
        4: B x C X H/(sclae*16) x W/(sclae*16)
        5: B x C X H/(sclae*16) x W/(sclae*16)

        known_matches is the positive matches sampled by dataloader.
        {'a':BxNx2,'b':BxNx2}
        '''

        self.max_size_x = F_a[0].shape[3]  # B x C x H x W
        self.max_size_y = F_a[0].shape[2]
 
        '''compute loss for each scale'''
        loss = 0
        tripletloss = 0
        gnloss = 0
        e1 = 0
        e2 = 0

        tripletloss_level = []
        gnloss_level = []
        loss_pos_mean_level = []
        loss_neg_mean_level = []

        N = positive_matches['a'].shape[1]  # the number of pos and neg matches
        scaling = [4*self.img_scale, 8*self.img_scale, 8*self.img_scale, 16*self.img_scale, 16*self.img_scale] # scaling w.r.t original size, i.e robotcar 1024*1024
        for i in range(len(F_a)):
            level = scaling[i]
            positive_matches_sampled = random_select_positive_matches(positive_matches['a'], positive_matches['b'], num_of_pairs=self.num_matches)
            fa_sliced_pos = extract_features(F_a[i], positive_matches_sampled['a'] / level)

            '''compute contrastive or triplet loss'''
            topM = np.clip(300*np.exp(-iteration*0.6/10000), a_min = 5, a_max=None)
            loss_triplet, loss_pos_mean, loss_neg_mean = self.pair_selector.get_triplets(F_a[i], F_b[i], positive_matches_sampled, level, topM = int(topM), dist_threshold=0.2, train_or_val=train_or_val, level=i)
            tripletloss_level.append(loss_triplet) # acquire tripletloss from different levels
            loss_pos_mean_level.append(loss_pos_mean)
            loss_neg_mean_level.append(loss_neg_mean)

            '''compute gn loss'''
            loss_gn_all = self.compute_gn_loss(fa_sliced_pos, F_b[i], positive_matches_sampled['b'] / level, train_or_val)
            loss_gn = loss_gn_all[0]
            e1 = e1 + loss_gn_all[1]
            e2 = e2 + loss_gn_all[2]
            gnloss_level.append(loss_gn) # acquire gnloss from different levels

            # if night_flag:
            #     loss = 4 * self.contrastive_lamda*loss_triplet + 4 * (self.gn_lamda * loss_gn) + loss 
            #     gnloss = 4 * (self.gn_lamda * loss_gn) + gnloss
            #     tripletloss = 4 * (self.contrastive_lamda * loss_triplet) + tripletloss
            # else:
            #     loss = self.contrastive_lamda*loss_triplet + (self.gn_lamda * loss_gn) + loss 
            #     gnloss = (self.gn_lamda * loss_gn) + gnloss
            #     tripletloss = (self.contrastive_lamda * loss_triplet) + tripletloss

            loss = self.contrastive_lamda*loss_triplet + (self.gn_lamda * loss_gn) + loss 
            gnloss = (self.gn_lamda * loss_gn) + gnloss
            tripletloss = (self.contrastive_lamda * loss_triplet) + tripletloss

        if night_flag:
            loss = loss 
            gnloss = gnloss
            tripletloss = tripletloss
        else:
            loss = 0.25 * loss 
            gnloss = 0.25 * gnloss
            tripletloss = 0.25 * tripletloss

        return loss, tripletloss, gnloss, tripletloss_level, gnloss_level, e1, e2, loss_pos_mean_level, loss_neg_mean_level