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
from model import PointNetfeat
from model import PointGenCon

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
    def __init__(self, in_features,out_features):
        self.conv = nn.Conv1d(in_features,out_features,1)
        self.bn = torch.nn.BatchNorm1d(out_features)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

def make(sizes,f,additional_args = None):
    return [f(*[sizes[i],sizes[i+1]]+ [] if additional_args is None or additional_args[i] is None else additional_args[i]) for i in range(len(sizes))]

from softpool_cuda.soft_pool_cuda_module import SoftPool3d

class GlobalTransform(nn.Module):
    def __init__(self, bottleneck_size, decoded_size):
        super(GlobalTransform, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.decoded_size = decoded_size
        sizes1 = [decoded_size, (decoded_size*decoded_size)//2, decoded_size*decoded_size]
        sizes2 = [decoded_size, decoded_size, 3]
        self.convs1 = nn.Sequential(*make(sizes1, lambda x,y : nn.Sequential(BatchNormConv1D(x,y)), SoftPool3d()))
        self.convs2 = nn.Sequential(*make(sizes2, lambda x,y : nn.Sequential(BatchNormConv1D(x,y)), SoftPool3d()))


    def forward(self, input, bottlenecked, decoded):
        bs = decoded.shape[0]
        x = self.convs(decoded)
        transform, _ = torch.max(x, 2).view(-1, self.decoded_size, self.decoded_size)
        transformed = torch.matmul(transform, decoded.view(bs, -1, self.decoded_size, 1)).view(decoded.shape)
        return self.convs2(transformed)




class TransformMSN(nn.Module):
    def __init__(self, encoder_state_dict, decoder_state_dict, num_points=8192, bottleneck_size=1024, n_primitives=16):
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
        self.encoder.load_state_dict(encoder_state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.decoder.load_state_dict(decoder_state_dict)
        self.decoder.conv4 = nn.Linear
        self.decoder.th = nn.Linear
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.global_transform = GlobalTransform(2 + self.bottleneck_size, self.decoder.conv3.out_features)
        self.res = Res()
        self.expansion = expansion.expansionPenaltyModule()

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