import torch
import torch.nn as nn 
import numpy as np
from utils import MyHardNegativePairSelector


class GNLoss(nn.Module):
    '''
    GN loss function.
    '''
    def __init__(self, margin = 1.0, lamda = 0.03): 
        super(GNLoss, self).__init__()
        self.margin = margin
        self.pair_selector = MyHardNegativePairSelector()
        self.lamda = lamda

    def compute_loss_pos(self, f_a, f_b, level, corres_pos):
        # size: D x H/level x W/level

        '''TODO: use bilinear interpolation'''
        # match_a_pos = corres_pos['a'] // (level*4) # the feature is scaled (divided by 4) to faster debugging
        # match_b_pos = corres_pos['b'] // (level*4)
        match_a_pos = corres_pos['a'] // (level)
        match_b_pos = corres_pos['b'] // (level)
        N_pos = match_a_pos.shape[1]  

        extracted_pos_fa = self.extract_features(f_a, match_a_pos)
        extracted_pos_fb = self.extract_features(f_b, match_a_pos)
        diff = extracted_pos_fa - extracted_pos_fb
        loss_pos = torch.sum(torch.pow(diff, 2) / N_pos) 

        # loss_pos = (diff**2).sum(-1)
        # loss_pos = loss_pos / N_pos

        return loss_pos

    def compute_loss_neg(self, f_a, f_b, level, corres_neg):
        # size: D x H/level x W/level

        '''TODO: use bilinear interpolation'''
        # match_a_neg = corres_neg['a'] // (level*4)
        # match_b_neg = corres_neg['b'] // (level*4)
        match_a_neg = corres_neg['a'] // (level)
        match_b_neg = corres_neg['b'] // (level)

        N_neg = match_a_neg.shape[1]

        extracted_neg_fa = self.extract_features(f_a, match_a_neg)
        extracted_neg_fb = self.extract_features(f_b, match_a_neg)
        diff = extracted_neg_fa - extracted_neg_fb

        mdist = self.margin - diff 
        mdist = torch.clamp(mdist, min=0.0)
        loss_neg = torch.sum(torch.pow(mdist, 2) / N_neg)

        return loss_neg

    def extract_features(self, f, indices):
        '''
        f: BxCxHxW
        indicies: BxNx2
        '''
        N = indices.shape[1]
        b, c, h, w = f.shape
        f_permuted = f.permute(1,0,2,3)
        f_2d = f_permuted.reshape((c, b*h*w))
        f_idx_2d = np.zeros((b*N))
        for b_th in range(b):
            m = indices[b_th]
            f_idx_2d[b_th*(N):(b_th+1)*N] = b_th*w*h + m[:,1]*w + m[:,0]
        f_idx_2d = torch.floor(torch.from_numpy(f_idx_2d)).type(torch.LongTensor).to(f_2d.device)
        return torch.index_select(f_2d, -1, f_idx_2d).transpose(0,1)
        # return torch.index_select(f_2d, -1, f_idx_2d)
        

    def compute_contrastive_loss(self, F_a, F_b, positive_pairs, negative_pairs):
        ''' compute pixel wise contrastive loss
            based on extracted feature pyramids.        
        '''
        loss= 0
        for i in range(len(F_a)):
            f_a = F_a[i]
            f_b = F_b[i]
            level = np.power(2, 3-i)
            # self.check_type_forward((f_a, f_b))
            # loss of positive pairs:
            loss_pos = self.compute_loss_pos(f_a, f_b, level = level, corres_pos = positive_pairs )
            # loss of negative pairs:
            loss_neg = self.compute_loss_neg(f_a, f_b, level = level, corres_neg = negative_pairs)
            loss = loss + (loss_pos + loss_neg)
        return loss
    
    '''TODO: for each scale'''
    def compute_gn_loss(self, out_a, out_b, corres_pos):
        '''Todo: 
        1. check all dims
        2. The gn loss scale is much larger than contrastive loss, i.e. gn loss = 7000, but contrastive loss = 200.
        please double check gn loss equations. 
        3. weighted sum of gn loss and contrastive loss?

        Notes:
        the feature is scaled (//4) to faster debugging,
        remember to fix this before train the network
        
        '''

        '''please refer to the gn net paper: page4, Algorithm 1'''
        # extract the target feature
        # ua = corres_pos['a']//4
        ua = corres_pos['a']
        f_t = self.extract_features(out_a, ua)
        # compute start point and its feature
        # ub = corres_pos['b']//4
        ub = corres_pos['b']
        B = ub.shape[0]
        N = ub.shape[1]
        xs = torch.round(torch.rand(ub.shape) + ub) # start at most 1 pixel away from u_b
        f_s = self.extract_features(out_b, xs)
        # compute residual
        r = f_s - f_t
        # compute Jacobian
        '''
        TODO: np.gradient
        '''
        f_s_gy, f_s_gx = np.gradient(out_b.detach().numpy(), axis=(2, 3)) # numerical derivative: J = d(out_b)/d(xs), feature gradients # to check if it is correct
        J_xs_x = self.extract_features(torch.from_numpy(f_s_gx), xs)
        J_xs_y = self.extract_features(torch.from_numpy(f_s_gy), xs)
        J = torch.stack([J_xs_x, J_xs_y], dim=-1) # todo: check dim
        # compute Heissian
        eps = 1e-9 # for invertibility, need to be smaller
        # TODO: create batched identity
        x = torch.eye(J.shape[2])
        x = x.reshape((1, J.shape[2], J.shape[2]))
        y = x.repeat(B*N, 1, 1)

        H = J.transpose(1,2) @ J + eps * y
        r = r.reshape((r.shape[0],r.shape[1],1))
        b = J.transpose(1,2) @ r
        xs_reshape = xs.reshape(B*N,2,1)
        miu = xs_reshape - torch.inverse(H) @ b
        # first error term
        e1 = 0.5 * ((ub.reshape(B*N,2,1) - miu).transpose(1,2)).type(torch.float32) @ H @ (ub.reshape(B*N,2,1) - miu).type(torch.float32) # check dim, very unsure
        e1 = torch.sum(e1)
        # second error term
        log_det = torch.log(torch.det(H))
        e2 = B*N * torch.log(torch.tensor(2 * np.pi)) - 0.5 * log_det.sum(-1) 
        e = e1 + e2
        return e


    def forward_debugging(self, F_a, F_b, match):
        '''
        F_a is a list containing 4 feature maps of different shapes
        1: B x C X H/8 x W/8
        2: B x C X H/4 x W/4
        3: B x C X H/2 x W/2
        4: B x C X H x W

        match is a list containing pos and neg matches
        match[0]: pos, {'a':BxNx2,'b':BxNx2}
        match[1]: neg, {'a':BxNx2,'b':BxNx2}
        '''

        '''compute constrastive loss'''
        loss_contras = self.compute_contrastive_loss(F_a, F_b, match[0], match[1])
        '''compute gn loss'''
        loss_gn = self.compute_gn_loss(F_a[-1], F_b[-1], match[0])
        '''compute weighted loss'''
        loss = self.lamda * loss_gn + loss_contras
        return loss
    
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

        '''get neg pairs, do it only for the finest feature'''
        fa_4 = F_a[-1]
        fb_4 = F_b[-1]
        B,N,_ = known_matches['a'].shape
        C = fa_4.shape[1]
        # slice features for positive matches
        fa_4_extracted = self.extract_features(fa_4, known_matches['a'])
        fa_4_extracted = fa_4_extracted.reshape((B,N,C))
        fb_4_extracted = self.extract_features(fb_4, known_matches['b'])
        fb_4_extracted = fb_4_extracted.reshape((B,N,C))
        # get hard negative samples
        negative_matches = self.pair_selector.get_pairs(fa_4_extracted, fb_4_extracted, known_matches['a'])
        '''compute constrastive loss'''
        loss_contras = self.compute_contrastive_loss(F_a, F_b, known_matches, negative_matches)
        '''compute gn loss'''
        loss_gn = self.compute_gn_loss(F_a[-1], F_b[-1], known_matches)
        '''compute weighted loss'''
        loss = self.lamda * loss_gn + loss_contras
        return loss




