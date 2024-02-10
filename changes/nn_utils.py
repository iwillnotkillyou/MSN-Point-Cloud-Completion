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
    def __init__(self, in_channels, out_channels, output_size, bias=False, kernel_size=1, stride=1):
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

    def forward(self, x):
        x = x.unfold(1, self.kernel_size, self.stride)
        out = torch.matmul(x.unsqueeze(2), self.weight).squeeze()
        if self.bias is not None:
            out += self.bias
        return out


class SimpleLocallyConnected1d():
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
    def __init__(self, in_channels, out_channels, output_size, kernel_size=1, stride=1):
        super().__init__()
        self.conv = LocallyConnected1d(in_channels, out_channels, output_size, False, kernel_size, stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BatchNormConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(BatchNormConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BatchNormConv1DTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(BatchNormConv1DTransformer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, _, __, x):
        return F.relu(self.bn(self.conv(x)))


def make(sizes, f, additional_args=None):
    return [f(*(
        [sizes[i], sizes[i + 1]] + [] if additional_args is None or additional_args[i] is None else additional_args[i]))
            for i in range(len(sizes) - 1)]
