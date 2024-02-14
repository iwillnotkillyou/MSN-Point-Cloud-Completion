import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
import open3d


# from utils import *

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]

def loadSplit(list_path):
    with open(os.path.join(list_path)) as file:
        return [line.strip() for line in file]

def saveSplit(list_path, model_list):
    with open(os.path.join(list_path)) as file:
        file.writelines([model_id for model_id in model_list])

class ShapeNet(data.Dataset):
    def __init__(self, list_path, npoints=8192):
        self.list_path = list_path
        self.npoints = npoints
        self.model_list = loadSplit(self.list_path)
        random.shuffle(self.model_list)
        self.len = len(self.model_list * 50)

    def __getitem__(self, index):
        model_id = self.model_list[index // 50].replace('/', '_')
        scan_id = index % 50

        def read_pcd(filename):
            pcd = open3d.io.read_point_cloud(filename)
            return torch.from_numpy(np.array(pcd.points)).float()

        partial = read_pcd(os.path.join("./data/partial", model_id + '_%d_denoised.pcd' % scan_id))
        complete = read_pcd(os.path.join("./data/complete/", '%s.pcd' % model_id))
        return model_id, resample_pcd(partial, 5000), resample_pcd(complete, self.npoints)

    def __len__(self):
        return self.len


class EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, folder, embedder_batch_size, transform=None):
        self.folder = folder
        self.files = os.listdir(folder)
        self.transform = transform
        self.embedder_batch_size = embedder_batch_size

    def __len__(self):
        return len(self.files * self.embedder_batch_size)

    def __getitem__(self, idx):
        id, x, target = torch.load(f"{self.folder}/{self.files[idx // self.embedder_batch_size]}")
        i = idx % self.embedder_batch_size
        try:
            r = id[i], (x[0][i], x[1][i]), target[i]
        except:
            print(x[0].shape, x[1].shape)
        return r
