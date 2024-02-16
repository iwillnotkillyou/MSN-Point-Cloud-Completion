from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from model import *


class LinearBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                               torch.nn.BatchNorm1d(out_channels),
                               torch.nn.ReLU())

    def forward(self, x):
        return self.m(x)

def channel_split(channels, num_splits, i):
    return channels // num_splits + np.max(0, np.sign((channels-num_splits) - i))

class DepthWise(nn.Module):

    def __init__(self, in_channels, out_channels, num_splits, modulef):
        super().__init__()
        self.m = nn.ModuleList([nn.Sequential(modulef(channel_split(in_channels,num_splits,i),
                                                     channel_split(out_channels,num_splits,i))) for i in
                                range(num_splits)])
        self.model_in_size_cum_sum = np.concatenate([[0], np.cumsum([x.in_features for x in self.m])])
        assert self.model_in_size_cum_sum[-1] == in_channels

    def forward(self, x):
        return torch.cat([self.m[i](x[:, self.model_size_cum_sum[i]: self.model_in_size_cum_sum[i+1]])
                          for i in range(self.num_splits)], 1)

class LinearDepthSepBN(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super().__init__()
        self.m = nn.ModuleList([nn.Sequential(torch.nn.Linear(in_channels//num_splits, out_channels//num_splits),
                               torch.nn.BatchNorm1d(out_channels//num_splits)) for _ in range(num_splits)])


    def forward(self, x):
        return self.m(x)


class LinearBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                               torch.nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return self.m(x)


class SimpleLocallyConnected1d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, bias=False):
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

    def forward(self, x):
        out = torch.matmul(x.unsqueeze(2), self.weight).squeeze()
        if self.bias is not None:
            out += self.bias
        return out


class BatchNormLocalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, output_size):
        super().__init__()
        self.conv = SimpleLocallyConnected1d(in_channels, out_channels, output_size, False)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class BatchNormConv1DNoAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class BatchNormConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


def DepthSepPool():
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def make(sizes, f, additional_args=None):
    return [f(*(
        [sizes[i], sizes[i + 1]] + [] if additional_args is None or additional_args[i] is None else additional_args[i]))
            for i in range(len(sizes) - 1)]
