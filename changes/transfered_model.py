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
    def __init__(self, additional_encoderf, encoder_modiff):
        super().__init__()
        self.num_points = 8192
        self.bottleneck_size = 1024
        self.n_primitives = 16
        self.additional_encoder = additional_encoderf()
        self.encoder_modif = encoder_modiff()
        self.encoder = nn.Sequential(
            PointNetfeatReturn2(self.num_points, self.encoder_modif),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

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
            outs.append(self.decoder[i](y))

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

class TransformMSNComplex(nn.Module):
    def __init__(self, residual, additional_encoder, additional_decoder, transform2d,
                 num_points=8192):
        super().__init__()
        self.num_points = num_points
        self.bottleneck_size = 1024
        self.n_primitives = 16
        self.encoder = nn.Sequential(
            PointNetfeat(num_points),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.transform2d = transform2d
        self.decoder = nn.ModuleList(
            [PointGenCon(transformf=self.transformDecoder,bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.additional_encoder = additional_encoder(self.bottleneck_size)
        self.additional_decoder = additional_decoder(3, (self.bottleneck_size+2)//2, (self.bottleneck_size+2)//4, self.bottleneck_size)
        self.res = residual()
        self.expansion = expansion.expansionPenaltyModule()

    def transformDecoder(self, x, mem):
        if mem == None:
            return x,(1,)
        elif mem[0] == 1:
            return x,(2,)
        elif mem[0] == 2:
            return x,(3,)
        elif mem[0] == 3:
            return x

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.encoder[0].parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.res.freeze()

    def forward(self, x):
        partial = x
        x = self.encoder[0](x)
        x = self.additional_encoder(partial, x)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        outs = []
        outs1 = []
        for i in range(0, self.n_primitives):
          rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
          rand_grid.data.uniform_(0, 1)
          y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
          self.transform2d()
          y = torch.cat((rand_grid, y), 1).contiguous()
          x0,x1 = self.decoder[i](y)
          outs.append(x0)
          outs1.append(x1)

        outs = torch.cat(outs, 2)
        outs1 = torch.cat(outs1, 2)
        outs = self.additional_decoder(outs1,outs,x)
        out1 = outs.transpose(1, 2).contiguous()
        dist, _, mean_mst_dis = self.expansion(out1, self.num_points // self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)
        id0 = torch.zeros(outs.shape[0], 1, outs.shape[2], device=torch.device('cuda')).contiguous()
        outs = torch.cat((outs, id0), 1)
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2], device=torch.device('cuda')).contiguous()
        partial = torch.cat((partial, id1), 1)
        xx = torch.cat((outs, partial), 2)

        resampled_idx = MDS_module.minimum_density_sample(xx[:, 0:3, :].transpose(1, 2).contiguous(), out1.shape[1],mean_mst_dis)
        xx = MDS_module.gather_operation(xx, resampled_idx)
        delta = self.res(xx)
        xx = xx[:, 0:3, :]
        out2 = (xx + delta).transpose(2, 1).contiguous()
        return out1, out2, loss_mst