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

class GlobalTransform(nn.Module):
    def __init__(self, bottleneck_size, partial_size, sizes, latents):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes) + (self.latents*self.latents,)
        self.convs = nn.Sequential(*make(sizes, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))
        self.register_buffer('identity', torch.diag(torch.ones(self.latents)))


    def forward(self, partial, x):
        bs = x.shape[0]
        x = x.view(bs,-1,self.latents).contiguous()
        outs = []
        transform_pre = self.convs(partial)
        softmaxweights = F.softmax(transform_pre,2)
        transform = (softmaxweights*transform_pre).sum(2).view(-1, self.latents, self.latents)
        identity = torch.broadcast_to(self.identity.unsqueeze(0),(bs, self.identity.shape[0], self.identity.shape[1]))
        transform = transform + identity
        for i in range(x.shape[1]):
          outs.append(torch.matmul(transform, x[:,i,:].unsqueeze(2)).squeeze())
        return torch.cat(outs,1)

class GlobalTransformGeneral(nn.Module):
    def __init__(self, bottleneck_size, partial_size, globalvsize, sizes, latents, use_globalv = True):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes) + (self.latents*self.latents,)
        self.convs = nn.Sequential(*make(sizes, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))
        self.register_buffer('identity', torch.diag(torch.ones(self.latents)))
        self.use_globalv = use_globalv
        if self.use_globalv:
          self.fcs = nn.Sequential(*make([globalvsize,16,self.latents*self.latents],
                                       lambda x,y : nn.Sequential(torch.nn.Linear(x,y),
                                                                  torch.nn.BatchNorm1d(y),
                                                                  torch.nn.ReLU())))


    def forward(self, partial, x, globalv):
        if len(x.shape) == 2:
          x = x.unsqueeze(2)
        bs = x.shape[0]
        x = x.view(bs,x.shape[1]//self.latents,self.latents,x.shape[2]).contiguous()
        outs = []
        transform_pre = self.convs(partial)
        softmaxweights = F.softmax(transform_pre,2)
        transform = (softmaxweights*transform_pre).sum(2)
        if self.use_globalv:
          transform = transform + self.fcs(globalv)
        transform = transform.view(-1, self.latents, self.latents)
        identity = torch.broadcast_to(self.identity.unsqueeze(0),(bs, self.identity.shape[0], self.identity.shape[1]))
        transform = transform + identity
        for i in range(x.shape[1]):
          outs.append(torch.matmul(transform, x[:,i,:,:]))
        return torch.cat(outs,1)

class GlobalTransformAdditive(nn.Module):
    def __init__(self, bottleneck_size, partial_size, sizes, latents):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes) + (bottleneck_size,)
        self.convs = nn.Sequential(*make(sizes, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))

    def forward(self, partial, x):
      v = self.convs(partial)
      softmaxweights = F.softmax(v,2)
      v = (softmaxweights*v).sum(2)
      return v + x

class GlobalTransformAdditiveGeneral(nn.Module):
    def __init__(self, bottleneck_size, partial_size, globalvsize, sizes, latents):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes) + (bottleneck_size,)
        self.convs = nn.Sequential(*make(sizes, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))
        self.usefcs = True
        if self.usefcs:
          self.fcs = nn.Sequential(*make([globalvsize,bottleneck_size*bottleneck_size,bottleneck_size],
                                       lambda x,y : nn.Sequential(torch.nn.Linear(x,y),
                                                                  torch.nn.BatchNorm1d(y),
                                                                  torch.nn.ReLU())))

    def forward(self, partial, x, globalv):
      v = self.convs(partial)
      softmaxweights = F.softmax(v,2)
      v = (softmaxweights*v).sum(2)
      if self.usefcs:
        v = v + self.fcs(globalv)
      os = x.shape
      if len(x.shape) < 2:
        x = x.unsqueeze(2)
      return (v.unsqueeze(2).broadcast_to(x.shape) + x).reshape(os)

class GlobalFeatureTransformPointnetAdditive(nn.Module):
    def __init__(self, bottleneck_size, partial_size, sizes, latents):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.latents = latents
        sizes1 = (partial_size,) + tuple(sizes[:-1])
        sizes2 = tuple(sizes[-2:]) + (bottleneck_size,)
        self.convs1 = nn.Sequential(*make(sizes1, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))
        self.convs2 = nn.Sequential(*make(sizes2, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))
        self.gt = GlobalTransformGeneral(sizes[-2], sizes[-2], None, (sizes[-2]//2,), sizes[-2]//2, use_globalv = False)


    def forward(self, partial, x):
      v = self.convs1(partial)
      v = self.gt(v,v,None)
      v = self.convs2(v)
      softmaxweights = F.softmax(v,2)
      v = (softmaxweights*v).sum(2)
      return v + x


class GlobalTransformIndentity(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, partial, x):
        return x

class GlobalTransformIndentityGlobalV(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, partial, x, globalv):
        return x



class TransformMSN(nn.Module):
    def __init__(self, residual, additional_encoder1, additional_encoder2, additional_decoder,
                 num_points=8192, n_primitives=16):
        super().__init__()
        self.num_points = num_points
        self.bottleneck_size = 1024
        self.n_primitives = n_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenConAppplyTransform(transformf=self.transformDecoder,bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.additional_encoder1 = additional_encoder1(self.bottleneck_size,3)
        self.additional_encoder2 = additional_encoder2(self.bottleneck_size, 3)
        self.additional_decoder = additional_decoder(3,(self.bottleneck_size+2)//4,self.bottleneck_size)
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
        x = self.additional_encoder1(partial, x)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        x = self.additional_encoder2(partial, x)
        outs = []
        outs1 = []
        for i in range(0, self.n_primitives):
          rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
          rand_grid.data.uniform_(0, 1)
          y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
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