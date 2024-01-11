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
        super(LocallyConnected1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, output_size, kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size)
            )
        else:
            self.register_parameter('bias', None)
    def forward(self,x):
        raise NotImplementedError()
        x = x.unfold(2, self.kernel_size, self.stride)
        out = torch.matmul(x,self.weight)
        if self.bias is not None:
            out += self.bias
        return out

class SimpleLocallyConnected1d():
    def __init__(self, in_channels,out_channels,output_size, bias = False):
        super(SimpleLocallyConnected1d, self).__init__()
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

class ResSmall(nn.Module):
    def __init__(self):
        super(ResSmall, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv4 = torch.nn.Conv1d(576, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 128, 1)
        self.conv6 = torch.nn.Conv1d(128, 3, 1)


        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.bn6 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 512)
        x = x.view(-1, 512, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.th(self.bn6(self.conv6(x)))
        return x

class Res(nn.Module):
    def __init__(self):
        super(Res, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)


        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x

class BatchNormConv1D(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size = 1, stride = 1):
        super(BatchNormConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

def make(sizes,f,additional_args = None):
    return [f(*([sizes[i],sizes[i+1]]+ [] if additional_args is None or additional_args[i] is None else additional_args[i])) for i in range(len(sizes)-1)]

from softpool_cuda.soft_pool_cuda_module import soft_pool1d, SoftPool1d

class GlobalTransform(nn.Module):
    def __init__(self, bottleneck_size, decoded_size):
        super(GlobalTransform, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.decoded_size = decoded_size
        self.latents = 50
        sizes0 = [decoded_size, self.latents]
        self.convs0 = nn.Sequential(*make(sizes0, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))
        sizes1 = [self.latents, self.latents*self.latents]
        sizes2 = [self.latents, 3]
        self.convs1 = nn.Sequential(*make(sizes1, lambda x,y : nn.Sequential(BatchNormConv1D(x,y), nn.MaxPool1d(2))))
        self.convs2 = nn.Sequential(*make(sizes2, lambda x,y : nn.Sequential(BatchNormConv1D(x,y))))
        self.identity = torch.diag(torch.ones(self.latents))


    def forward(self, partial, bottlenecked, decoded):
        bs = decoded.shape[0]
        x = self.convs0(decoded)
        transform_pre = self.convs1(x)
        softmaxweights = F.softmax(transform_pre,2)
        identity = self.identity.repeat(bs, 1)
        transform = (softmaxweights*transform_pre).view(-1, self.latents, self.latents).contiguous() + identity
        x = torch.matmul(transform, x.transpose(0, 1).transpose(0,2).unsqueeze(3))
        x = x.squeeze().transpose(0, 2).transpose(0, 1).contiguous()
        return self.convs2(x)




class TransformMSN(nn.Module):
    def __init__(self, res, num_points=8192, bottleneck_size=1024, n_primitives=16):
        super(TransformMSN, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        for de in self.decoder:
          de.conv4 = nn.Identity()
          de.th = nn.Identity()
        print(self.decoder[0].conv3.out_channels)
        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        self.global_transform = GlobalTransform(2 + self.bottleneck_size, self.decoder[0].conv3.out_channels) + identity
        self.res = res
        self.expansion = expansion.expansionPenaltyModule()

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        partial = x
        x = self.encoder(x)
        outs = []
        for i in range(0, self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))


        outs = torch.cat(outs, 2)
        outs = self.global_transform(partial,x,outs).contiguous()
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