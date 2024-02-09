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

class LocallyConnected1d():
    def __init__(self, in_channels,out_channels,output_size, bias = False, kernel_size = 1, stride = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(
            torch.randn(output_size, out_channels, in_channels * kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(out_channels, output_size)
            )
        else:
            self.register_parameter('bias', None)
    def forward(self,x):
        x = x.unfold(1, self.kernel_size, self.stride)
        out = torch.matmul(x.unsqueeze(2), self.weight).squeeze()
        if self.bias is not None:
            out += self.bias
        return out

class SimpleLocallyConnected1d():
    def __init__(self, in_channels,out_channels,output_size, bias = False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(output_size, out_channels, in_channels)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(output_size, out_channels)
            )
        else:
            self.register_parameter('bias', None)
    def forward(self,x):
        out = torch.matmul(x.unsqueeze(2),self.weight).squeeze()
        if self.bias is not None:
            out += self.bias
        return out

class ResSizes(nn.Module):
    def __init__(self, sizesup, sizesdown):
        super().__init__()


        lsup = []
        for i in range(len(sizesup)-1):
            lsup.append(BatchNormConv1D(sizesup[i],sizesup[i+1]))
        lsdown = []
        for i in range(len(sizesdown)-1):
            lsdown.append(BatchNormConv1D(sizesdown[i] + (0 if i != 0 else sizesup[0]),sizesdown[i+1]))
        self.sizesdown = sizesdown
        self.sizesup = sizesup
        self.convsup = torch.nn.Sequential(*lsup)
        self.convsdown = torch.nn.Sequential(*lsdown)
        self.conv1 = BatchNormConv1D(4, sizesup[0])
        self.convlast = torch.nn.Conv1d(sizesdown[-1], 3, 1)
        self.bnlast = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = self.conv1(x)
        pointfeat = x
        x = self.convsup(x)
        s = x.size()[1]
        softmaxweights = F.softmax(x,2)
        x = (softmaxweights*x).sum(2)
        x = x.view(batchsize, -1, 1)
        x = x.broadcast_to(x.shape[0], x.shape[1], npoints)
        x = torch.cat([x, pointfeat], 1)
        x = self.convsdown(x)
        x = self.th(self.bnlast(self.convlast(x)))
        return x

class BatchNormLocalConv1D(nn.Module):
    def __init__(self, in_channels,out_channels,output_size,kernel_size = 1, stride = 1):
        super().__init__()
        self.conv = LocallyConnected1d(in_channels,out_channels,output_size,False,kernel_size,stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class BatchNormConv1D(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size = 1, stride = 1):
        super(BatchNormConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class BatchNormConv1DTransformer(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size = 1, stride = 1):
        super(BatchNormConv1DTransformer, self).__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, _, __, x):
        return F.relu(self.bn(self.conv(x)))

def make(sizes,f,additional_args = None):
    return [f(*([sizes[i],sizes[i+1]]+ [] if additional_args is None or additional_args[i] is None else additional_args[i])) for i in range(len(sizes)-1)]

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

class PointGenConReturnsTwoLast(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super().__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        saved = x
        x = self.th(self.conv4(x))
        return x, saved

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
    def __init__(self, residual, additional_encoder, additional_decoder,
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
            [PointGenConReturnsTwoLast(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.additional_encoder = additional_encoder(self.bottleneck_size,3)
        self.additional_decoder = additional_decoder(3,(self.bottleneck_size+2)//4,self.bottleneck_size)
        self.res = residual()
        self.expansion = expansion.expansionPenaltyModule()

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.res.freeze()

    def forward(self, x):
        partial = x
        x = self.encoder(x)
        x = self.additional_encoder(partial,x)
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