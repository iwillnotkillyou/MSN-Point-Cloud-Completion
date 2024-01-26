import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from model import *
from utils import *
import os
import json
import time, datetime
from time import time
from torch.utils.cpp_extension import load
import emd.emd_module as emd
from tqdm import tqdm
class FullModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, embeddings, gt, eps, iters):
        output1, output2, expansion_penalty = self.model(inputs,embeddings)
        gt = gt[:, :, :3]

        dist, _ = self.EMD(output1, gt, eps, iters)
        emd1 = torch.sqrt(dist).mean(1).contiguous()

        dist, _ = self.EMD(output2, gt, eps, iters)
        emd2 = torch.sqrt(dist).mean(1).contiguous()

        return output1, output2, emd1, emd2, expansion_penalty

torch.cuda.empty_cache()
import gc
gc.collect()
class opt:
  def __init__(self):
    self.batchSize = 8
    self.workers = 1
    self.nepoch = 100
    self.epoch_iter_limit = 320
    self.model = ''
    self.num_points = 8192
    self.n_primitives = 16
    self.env = "MSN_TRAIN"
    self.perc_data = 0.001
    self.perc_val_data = 0.0005
    self.manualSeed = 0
    self.epoch_iter_limit_val = 16
    self.run_embedder = True
    self.embeddingpath = "/content/embeddings"
    self.embed_nepoch = 1

default_opt = opt()

def make_embedder(args = default_opt):
  model = TransformMSNEmbedder()
  #model = torch.nn.DataParallel(model)
  print('modelf',args.model)
  if args.model != '':
      previous_state_dict = torch.load(args.model)
      model.load_state_dict(previous_state_dict,False,True)
      print("Previous weight loaded ")
      if False:
        sd = model.state_dict()
        for w in sd:
          if w in previous_state_dict:
            print(w)
            print(torch.equal(sd[w], previous_state_dict[w]))
  model.freeze()
  return model

def make_model(in_dims, args = default_opt):
  def gtFunc(x,y,z):
    return BatchNormConv1DTransformer(y,z)
  def residualFunc(x):
    return x
  model = TransformMSNNoEmbedder(residualFunc = residualFunc, gtFunc = gtFunc,in_dims = in_dims)
  network = FullModel(model)
  #network = torch.nn.DataParallel(network)
  print('modelf',args.model)
  #model.apply(weights_init) #initialization of the weight
  if args.model != '':
      previous_state_dict = torch.load(args.model)
      model.load_state_dict(previous_state_dict,False,True)
      print("Previous weight loaded ")
      if False:
        sd = model.state_dict()
        for w in sd:
          if w in previous_state_dict:
            print(w)
            print(torch.equal(sd[w], previous_state_dict[w]))
  model.freeze()
  return network

import shutil
def embed(trainp,valp,embedder_network,args):
  embedder_network.to("cuda")
  shutil.rmtree(trainp)
  shutil.rmtree(valp)
  os.makedirs(trainp, exist_ok = True)
  os.makedirs(valp, exist_ok = True)
  embedder_network.eval()
  x = ShapeNet(train=False, npoints=args.num_points)
  perc_train = (1 - args.perc_val_data)*args.perc_data
  perc_val = args.perc_val_data*args.perc_data
  embedder_dataset, embedder_dataset_val, _ = torch.utils.data.random_split(x,[perc_train,perc_val,1-(perc_train+perc_val)])
  embedder_dataloader = torch.utils.data.DataLoader(embedder_dataset, batch_size=args.embed_batch_size,
                                            shuffle=False, num_workers = args.workers,drop_last=True)
  embedder_dataloader_val = torch.utils.data.DataLoader(embedder_dataset_val, batch_size=args.embed_batch_size,
                                            shuffle=False, num_workers = args.workers, drop_last=True)
  for epoch in range(args.embed_nepoch):

    def f(loader,path):
      with torch.no_grad():
        for i, data in tqdm(enumerate(loader),f"making embeddings {path}", len(loader)):
          percd = int(100 * i / len(loader))
          id, input, gt = data
          input = input.transpose(1,2).float().cuda()
          gt = gt.float().cuda()
          torch.save((id,embedder_network(input),gt),f"{path}{i}")
    f(embedder_dataloader,f"{trainp}/{epoch}_")
    f(embedder_dataloader_val,f"{valp}/{epoch}_")
  del(embedder_dataset)
  del(embedder_dataset_val)

def train(embedder_network, network, dir_name, args = default_opt):
  lrate = 0.001 #learning rate
  optimizer = optim.Adam(model.parameters(), lr = lrate)
  train_curve = []
  val_curve = []
  labels_generated_points = (torch.Tensor(range(1, (args.n_primitives+1)*(args.num_points//args.n_primitives)+1))
  .view(args.num_points//args.n_primitives,(args.n_primitives+1)).transpose(0,1))
  labels_generated_points = (labels_generated_points)%(args.n_primitives+1)
  labels_generated_points = labels_generated_points.contiguous().view(-1)
  print("Random Seed: ", args.manualSeed)
  random.seed(args.manualSeed)
  torch.manual_seed(args.manualSeed)
  best_val_loss = 10

  trainp = f"{args.embeddingpath}/train"
  valp = f"{args.embeddingpath}/val"
  os.makedirs(trainp, exist_ok = True)
  os.makedirs(valp, exist_ok = True)

  if args.embed_nepoch is not None:
      embed(trainp,valp,embedder_network,args)
  del(embedder_network)
  network.to("cuda")
  torch.cuda.empty_cache()


  dataset = EmbeddingsDataset(trainp, args.embed_batch_size)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                                shuffle=True)
  dataset_val = EmbeddingsDataset(valp, args.embed_batch_size)
  dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batchSize,
                                              shuffle=False)

  len_dataset = len(dataset)
  len_val_dataset = len(dataset_val)
  print("Train Set Size: ", len_dataset)
  try:
    for epoch in range(args.nepoch):
        train_loss = np.zeros(len(dataloader))
        val_loss = np.zeros(len(dataloader_val))
        #TRAIN MODE
        model.train()

        # learning rate schedule
        if epoch==10:
            optimizer = optim.Adam(model.parameters(), lr = lrate/10)
        if epoch==20:
            optimizer = optim.Adam(model.parameters(), lr = lrate/100)
        if epoch==30:
            optimizer = optim.Adam(model.parameters(), lr = lrate/1000)

        for i, data in enumerate(dataloader, 0):
            if args.epoch_iter_limit is not None and i > args.epoch_iter_limit:
              break
            optimizer.zero_grad()
            id, (partial,input), gt = data
            input = input.float().cuda()
            partial = partial.float().cuda()
            gt = gt.float().cuda()

            output1, output2, emd1, emd2, expansion_penalty  = network(partial, input, gt.contiguous(), 0.005, 10)
            emd1m = emd1.mean()
            emd1mi = emd1m.item()
            emd2m = emd2.mean()
            emd2mi = emd2m.item()
            exppm = expansion_penalty.mean()
            exppmi = exppm.item()
            loss_net = emd1m + emd2m+ exppm * 0.1
            train_loss[i] = emd2mi
            loss_net.backward()
            optimizer.step()

            print(args.env + ' train [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f'
                  %(epoch, i, len_dataset/args.batchSize, emd1mi, emd2mi,
                    exppmi))
            if (i*args.batchSize) % 100 == 0:
              if not os.path.exists(dir_name):
                os.makedirs(dir_name)
              torch.save(model.state_dict(), '%s/network.pth' % (dir_name))
            del(input)
            del(partial)
        train_curve.append(np.mean(train_loss))
        # VALIDATION
        if True:
            emds = []
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(dataloader_val):
                    if args.epoch_iter_limit_val is not None and i > args.epoch_iter_limit_val:
                      break
                    id, (partial,input), gt = data
                    input = input.float().cuda()
                    partial = partial.float().cuda()
                    gt = gt.float().cuda()
                    output1, output2, emd1, emd2, expansion_penalty  = network(partial, input, gt.contiguous(), 0.004, 20)
                    emd1m = emd1.mean()
                    emd1mi = emd1m.item()
                    emd2m = emd2.mean()
                    emd2mi = emd2m.item()
                    val_loss[i] = emd2mi
                    best_val_loss = max(best_val_loss,emd2mi)
                    exppm = expansion_penalty.mean()
                    exppmi = exppm.item()
                    idx = random.randint(0, input.size()[0] - 1)
                    print(args.env + ' val [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f'
                          %(epoch, i, len_val_dataset/args.batchSize, emd1mi,
                            emd2mi, exppmi))
                    del(input)
                    del(partial)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            logname = os.path.join(dir_name, 'log.txt')
            val_curve.append(np.mean(val_loss))
            log_table = {
              "train_loss" : np.mean(train_loss),
              "val_loss" : np.mean(val_loss),
              "epoch" : epoch,
              "lr" : lrate,
              "bestval" : best_val_loss,

            }
            print(log_table)
            with open(logname, 'a') as f:
                f.write('json_stats: ' + json.dumps(log_table) + '\n')

            print('saving net...')
            torch.save(model.state_dict(), '%s/network.pth' % (dir_name))
  except:
    del(dataset)
    del(dataloader)
    del(dataset_val)
    del(dataloader_val)
    raise


def cm1(model):
  for n, m in model.named_children():
    print(n,m)
  return model
from copy import copy
now = datetime.datetime.now()
save_path = "/content/drive/MyDrive/saved/model" + now.isoformat() if False else "/content/saved/new"
args = copy(default_opt)
args.manualSeed = random.randint(1, 10000)
args.base_model = './trained_model/network.pth'
args.model = '/content/saved/network.pth' if False else None
should_embed = first
args.embed_nepoch = 1 if should_embed else None
args.embed_batch_size = 1
args.epoch_iter_limit_val = None
args.batchSize = 16
train = False
args.epoch_iter_limit = None
args.perc_data = 0.1 if train else 1
args.perc_val_data = 0.1 if train else 0.999
args.embeddingpath = "/content/drive/MyDrive/embeddings" if False else "/content/embeddings"
print(args.embed_nepoch)
print(args.model)
try:
  del(embedder)
except:
  pass
try:
  del(model)
except:
  pass

if True:
  embedder = make_embedder(args) if should_embed else None
  in_c = embedder.decoder[0].conv3.out_channels if should_embed else 256
  print(in_c)
  #args.model = "/content/drive/MyDrive/saved/model.py"
  model = make_model(in_c, args)
  cs = 0
  for name, param in model.named_parameters():
    if param.requires_grad:
      cs += param.nelement()
      print(name, cs)
  train(embedder, model,save_path, args)