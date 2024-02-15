from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
import expansion_penalty.expansion_penalty_module as expansion
import MDS.MDS_module as MDS_module
from model import *
from res import *
from global_transform import *


class TransformMSN(nn.Module):
    def __init__(self, additional_encoderf, pointnetfeatf, additionaldecoderf = None, num_points = 8092, train_encoder = False):
        super().__init__()
        self.num_points = num_points
        self.bottleneck_size = 1024
        self.n_primitives = 16
        self.additional_encoder = additional_encoderf()
        self.encoder = nn.Sequential(
            pointnetfeatf(),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.train_encoder = train_encoder
        self.additionaldecoder = None if additionaldecoderf is None else [additionaldecoderf()
                                                                          for i in range(self.n_primitives)]
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

    def freeze(self):
        if not self.train_encoder:
            self.encoder.freeze()
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.res.parameters():
            param.requires_grad = False

    def changed_state_dict(self):
        d = self.state_dict()
        ks = ['additional_encoder', 'encoder_modif']
        r = dict((k, d[k]) for k in d if any(k2 in k for k2 in ks))
        return r

    def forward(self, x):
        partial = x
        x, x1 = self.encoder[0](x)
        saved = x
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        x = self.additional_encoder(partial, x1, saved, x)
        outs = []
        for i in range(0, self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            y = self.decoder[i](y)
            y = self.additionaldecoder[i](y, y, x)
            outs.append(y)

        outs = torch.cat(outs, 2).contiguous()
        out1 = outs.transpose(1, 2).contiguous()

        dist, _, mean_mst_dis = self.expansion(out1, self.num_points // self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)

        id0 = torch.zeros(outs.shape[0], 1, outs.shape[2]).cuda().contiguous()
        outs = torch.cat((outs, id0), 1)
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2]).cuda().contiguous()
        partial = torch.cat((partial, id1), 1)
        xx = torch.cat((outs, partial), 2)

        resampled_idx = MDS_module.minimum_density_sample(xx[:, 0:3, :].transpose(1, 2).contiguous(), out1.shape[1],
                                                          mean_mst_dis)
        xx = MDS_module.gather_operation(xx, resampled_idx)
        delta = self.res(xx)
        xx = xx[:, 0:3, :]
        out2 = (xx + delta).transpose(2, 1).contiguous()
        return out1, out2, loss_mst

