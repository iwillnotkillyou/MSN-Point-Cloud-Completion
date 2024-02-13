import os
import torch
import time
import numpy as np
import json
import logging
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
import os
from torch import nn
from collections import defaultdict
import h5py
from multiprocessing import Queue
from data_process import kill_data_processes
from shapenet import ShapenetDataProcess
import subprocess
import sys

chamferdir = "../completion3d/pytorch/utils/chamfer"
from torch.utils.cpp_extension import load

chamfer = load(name="chamfer", sources=[f"{chamferdir}/chamfer_cuda.cpp", f"{chamferdir}/chamfer.cu"])


class chamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class chamferDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        return chamferFunction.apply(input1, input2)
