import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class hetero_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type='l2'):
        super(hetero_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.3)
        # self.w1 = nn.Parameter(torch.tensor(1).float().cuda())
        # self.w2 = nn.Parameter(torch.tensor(1).float().cuda())
    def forward(self, feat1, feat2, label1, label2):
        feats = torch.cat((feat1, feat2), dim=0)
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist_ = max(0, self.dist(center1, center2) - self.margin)
                else:
                    dist_ += max(0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist_ = max(0, 1 - self.dist(center1, center2) - self.margin)
                else:
                    dist_ += max(0, 1 - self.dist(center1, center2) - self.margin)
        label_uni = label1.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)

        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)
        m = 4
        center_v = inputs[:m, :]  # 4, 512
        center_t = inputs[m:, :]  # 4, 512
        dist_vv = pdist_torch(center_v, center_v)
        dist_tt = pdist_torch(center_t, center_t)
        dist_vt = pdist_torch(center_v, center_t)
        dist_tv = pdist_torch(center_t, center_v)
        mask = label_uni.expand(m, m).eq(label_uni.expand(m, m).t())
        #-------------------# inter-modality
        # P: v v  N: t

        dist_ap1, dist_an1 = [], []
        for i in range(m):
            dist_an1.append(dist_vv[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap1.append(dist_vt[i][mask[i]].max().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an1)
        loss1 = self.ranking_loss(dist_an1, dist_ap1, y)
        # P: t t  N: v
        dist_ap2, dist_an2 = [], []
        for i in range(m):
            dist_an2.append(dist_tt[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap2.append(dist_tv[i][mask[i]].max().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        # Compute ranking hinge loss
        loss2 = self.ranking_loss(dist_an2, dist_ap2, y)
        #-------------------#

        return dist_ + loss1 + loss2

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx
