import math

import torch
import torch.nn as nn

import torch.nn.functional as F


class GMul2(nn.Module):

    def __init__(self):
        super(GMul2, self).__init__()

    def forward(self, x, W):
        # x is a tensor of size (bs, p1, N, 1)
        # W is a tensor of size (bs, N, N, J)
        # Output: a tensor of size (bs, J*p1, N, 1)
        res = []
        for j in range(W.size(3)):
            res.append(
                torch.bmm(x.squeeze(3), W[:, :, :, j]))

        z = torch.cat(res, dim=1).unsqueeze(3)
        return z


class GNNAtomic(nn.Module):

    def __init__(self, featuremaps, J, last=False):
        super(GNNAtomic, self).__init__()
        self.featuremaps = featuremaps  # inp_c, mid_c, out_c
        self.J = J
        self.last = last
        self.gmul = GMul2()
        self.conv1 = nn.Conv2d(J * featuremaps[0], featuremaps[2], 1, 1)
        self.conv2 = nn.Conv2d(J * featuremaps[0], featuremaps[2], 1, 1)
        self.batch_norm = nn.BatchNorm2d(2 * featuremaps[2])

    def forward(self, x, W):
        # x is a tensor of size (bs, inp_c, N, 1)
        # W is a tensor of size (bs, N, N, J)
        x1 = self.gmul(x, W)  # x1: (bs, J*inp_c, N, 1)
        y1 = F.relu(self.conv1(x1))  # y1: (bs, out_c, N, 1)
        y2 = self.conv2(x1)  # y2: (bs, out_c, N, 1)

        # z: (bs, 2 * out_c, N, 1)
        z = self.batch_norm(torch.cat([y1, y2], dim=1))

        if not self.last:
            return z, W  # z: (bs, 2 * out_c, N, 1), (bs, N, N, J)
        else:
            return z, W  # z: (bs, 2 * out_c, N, 1), (bs, N, N, J)


class GNNMultiClass(nn.Module):

    def __init__(self, featuremaps, J, NClasses, N):
        super(GNNMultiClass, self).__init__()
        self.featuremaps = featuremaps  # inp_c, mid_c, out_c
        self.J = J
        self.NClasses = NClasses
        self.N = N
        self.gmul = GMul2()
        self.conv1 = nn.Conv2d(J * featuremaps[0], NClasses, 1, 1)

    def forward(self, x, W):
        # x is a tensor of size (bs, inp_c, N, 1)
        # W is a tensor of size (bs, N, N, J)
        x1 = self.gmul(x, W)  # x1: (bs, J*inp_c, N, 1)
        y1 = self.conv1(x1)   # y1: (bs, NClasses, N, 1)
        z = torch.transpose(y1.squeeze(), 1, 0)  # z: (bs, N, NClasses)
        return z  # or (N, NClasses) if bs==1 (usually, it is the case)


class IndexModule(nn.Module):

    def __init__(self, index):
        super(IndexModule, self).__init__()
        self.index = index

    def forward(self, x, i=None):
        if i is None:
            return x.index(self.index)
        else:
            return x.index(self.index)[i]


class GNNModular(nn.Module):

    def __init__(
            self,
            featuremap_in,   # n_in, n_mi, n_out
            featuremap_mi,   # 2 * n_out, n_mi_1, n_out
            featuremap_end,  # 2 * n_out, n_mi_2, n_out_2
            NLayers,
            J,
            NClasses,
            N):
        super(GNNModular, self).__init__()
        self.featuremap_in = featuremap_in
        self.featuremap_mi = featuremap_mi
        self.featuremap_end = featuremap_end
        self.NLayers = NLayers
        self.J = J
        self.NClasses = NClasses
        self.N = N

        listModules = []

        listModules.append(GNNAtomic(featuremap_in, 2 + J, False))
        for i in range(self.NLayers):
            listModules.append(GNNAtomic(featuremap_mi, 2 + J, False))

        listModules.append(GNNAtomic(featuremap_mi, 2 + J, True))
        listModules.append(GNNMultiClass(featuremap_end, 2 + J, NClasses, N))
        self.list_modules = nn.ModuleList(listModules)
        self._len = len(listModules)

    def forward(self, x, W):
        for i, l in enumerate(self.list_modules):
            if i < self._len - 1:
                x, W = l(x, W)

        z = self.list_modules[-1](x, W)
        return z  # z: (bs, N, NClasses) or (N, NClasses) if bs==1
