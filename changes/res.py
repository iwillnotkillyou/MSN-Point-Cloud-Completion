from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from model import *
from nn_utils import *


class PointNetResLastLayerSizes(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        sizes = (256,) + sizes
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.convs = nn.Sequential(*make(sizes, lambda x, y: BatchNormConv1D(x, y)))
        self.lastconv = torch.nn.Conv1d(sizes[-1], 3, 1)
        self.th = nn.Tanh()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.convs.parameters():
            param.requires_grad = True
        for param in self.lastconv.parameters():
            param.requires_grad = True

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.convs(x)
        x = self.th(self.lastconv(x))
        return x


class PointNetResSoftMax(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        sizes = (1088,) + tuple(sizes)
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.convs = nn.Sequential(*make(sizes, lambda x, y: BatchNormConv1D(x, y)))
        self.lastconv = torch.nn.Conv1d(sizes[-1], 3, 1)
        self.th = nn.Tanh()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.convs.parameters():
            param.requires_grad = True
        for param in self.lastconv.parameters():
            param.requires_grad = True

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        softmaxweights = F.softmax(x, 2)
        x = (softmaxweights * x).sum(2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = self.convs(x)
        x = self.th(self.lastconv(x))
        return x
class PointNetResFT(nn.Module):
    def __init__(self):
        super().__init__()
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

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.conv7.parameters():
            param.requires_grad = True
        for param in self.bn7.parameters():
            param.requires_grad = True

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x

class PointGenConAppplyTransform(nn.Module):
    def __init__(self, transformf, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super().__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
        self.transformf = transformf
        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class ResSizes(nn.Module):
    def __init__(self, sizesup, sizesdown):
        super().__init__()

        lsup = []
        for i in range(len(sizesup) - 1):
            lsup.append(BatchNormConv1D(sizesup[i], sizesup[i + 1]))
        lsdown = []
        for i in range(len(sizesdown) - 1):
            lsdown.append(BatchNormConv1D(sizesdown[i] + (0 if i != 0 else sizesup[0]), sizesdown[i + 1]))
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
        softmaxweights = F.softmax(x, 2)
        x = (softmaxweights * x).sum(2)
        x = x.view(batchsize, -1, 1)
        x = x.broadcast_to(x.shape[0], x.shape[1], npoints)
        x = torch.cat([x, pointfeat], 1)
        x = self.convsdown(x)
        x = self.th(self.bnlast(self.convlast(x)))
        return x

class PointNetfeatReturn2(nn.Module):
    def __init__(self, num_points, extra):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.extra = extra
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x128 = x
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x128 = self.extra(x128,x128,x)
        x = self.bn3(self.conv3(x128))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x, x128
