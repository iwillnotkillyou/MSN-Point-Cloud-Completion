import random
import torch.optim as optim
from transfered_model import *
from model import *
from utils import *
import os
import json
from dataset import ShapeNet
from my_chamfer_interface import chamferDist
import gc

class KFACargs:
    def __init__(self, momentum, cov_ema_decay, damping, stab_coeff, use_cholesky, adjust_momentum,
                 acc_iters):
        (self.momentum, self.cov_ema_decay,
         self.damping, self.stab_coeff,
         self.use_cholesky,
         self.adjust_momentum, self.acc_iters) = (momentum, cov_ema_decay,
                                                  damping, stab_coeff, use_cholesky,
                                                  adjust_momentum, acc_iters)
    def __str__(self):
        return str((self.momentum, self.cov_ema_decay,
         self.damping, self.stab_coeff,
         self.use_cholesky,
         self.adjust_momentum, self.acc_iters))

defaultKFACargs = KFACargs(0.80, 0.90, 0.01, 5.0,
                           True, True, 360)

def trainFull(network, dir_name, val_only, args, lrate = 0.001, kfacargs = defaultKFACargs):
  torch.save(network.model.changed_state_dict(), '%s/network.pth' % (dir_name))
  def optimf(lr):
    return optims.KFAC(lr, kfacargs.momentum, kfacargs.cov_ema_decay,
                        kfacargs.damping, kfacargs.stab_coeff,
                        use_cholesky=kfacargs.use_cholesky,
                        adjust_momentum=kfacargs.adjust_momentum)
  usefirstorder = args.usefirstorder
  print(usefirstorder)
  if not usefirstorder:
      try:
          import chainerkfac.optimizers as optims
          optimizer = optimf(lrate)
      except:
          usefirstorder = True
  if usefirstorder:
      optimizer = optim.Adam(network.parameters(), lr = lrate)
  train_curve = []
  val_curve = []
  train_curvecd = []
  val_curvecd = []
  labels_generated_points = (torch.Tensor(range(1, (args.n_primitives+1)*(args.num_points//args.n_primitives)+1))
  .view(args.num_points//args.n_primitives,(args.n_primitives+1)).transpose(0,1))
  labels_generated_points = (labels_generated_points)%(args.n_primitives+1)
  labels_generated_points = labels_generated_points.contiguous().view(-1)
  print("Random Seed: ", args.manualSeed)
  random.seed(args.manualSeed)
  torch.manual_seed(args.manualSeed)
  best_val_loss = np.inf
  network.to("cuda")
  x = ShapeNet(train=False, npoints=args.num_points)
  perc_train = (1 - args.perc_val_data)*args.perc_data
  perc_val = args.perc_val_data*args.perc_data
  dataset, dataset_val, _ = torch.utils.data.random_split(x,[perc_train,perc_val,1-(perc_train+perc_val)])
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers = args.workers,drop_last=True)
  dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batchSize,
                                            shuffle=False, num_workers = args.workers, drop_last=True)
  assert(len(dataset_val ) // args.batchSize > 0),f"{len(dataset_val )} // {args.batchSize} == 0"
  len_dataset = len(dataset)
  len_val_dataset = len(dataset_val)
  print("Train Set Size: ", len_dataset)
  try:
    for epoch in range(args.nepoch):
        train_loss = np.zeros(len(dataloader))
        val_loss = np.zeros(len(dataloader_val))
        train_losscd = np.zeros(len(dataloader))
        val_losscd = np.zeros(len(dataloader_val))
        #TRAIN MODE
        network.train()

        # learning rate schedule
        if usefirstorder:
          if epoch==3:
              optimizer = optim.Adam(network.parameters(), lr = lrate/10.0)
          if epoch==6:
              optimizer = optim.Adam(network.parameters(), lr = lrate/100.0)
        else:
              optimizer.lr = optimizer.lr*0.98


        if not val_only:
          for i, data in enumerate(dataloader, 0):
              if args.epoch_iter_limit is not None and i > args.epoch_iter_limit:
                break
              optimizer.zero_grad()
              id, input, gt = data
              input = input.float().cuda().transpose(1,2)
              gt = gt.float().cuda()

              output1, output2, emd1, emd2, expansion_penalty  = network(input, gt.contiguous(), 0.005, 10)
              emd1m = emd1.mean()
              emd1mi = emd1m.item()
              emd2m = emd2.mean()
              emd2mi = emd2m.item()
              exppm = expansion_penalty.mean()
              exppmi = exppm.item()
              loss_net = emd1m + emd2m+ exppm * 0.1
              dist1, dist2 = chamferDist()(output2.float(), gt)
              train_losscd[i] = torch.mean(dist2) + torch.mean(dist1)
              train_loss[i] = emd2mi
              loss_net.backward()
              optimizer.step()

              print(args.env + ' train [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f cd : %f'
                    %(epoch, i, len_dataset/args.batchSize, emd1mi, emd2mi,
                      exppmi, train_losscd[i]))
              del(input)
              del(output1)
              del(output2)
          train_curve.append(np.mean(train_loss))
          train_curvecd.append(np.mean(train_losscd))

        # VALIDATION
        if True:
            network.eval()
            with torch.no_grad():
                for i, data in enumerate(dataloader_val):
                    if args.epoch_iter_limit_val is not None and i > args.epoch_iter_limit_val:
                      break
                    id,input, gt = data
                    input = input.float().cuda().transpose(1,2)
                    gt = gt.float().cuda()
                    output1, output2, emd1, emd2, expansion_penalty  = network(input, gt.contiguous(), 0.004, 100 if val_only else 20)
                    emd1m = emd1.mean()
                    emd1mi = emd1m.item()
                    emd2m = emd2.mean()
                    emd2mi = emd2m.item()
                    dist1, dist2 = chamferDist()(output2.float(), gt)
                    val_losscd[i] = (torch.mean(dist2) + torch.mean(dist1)).item()
                    val_loss[i] = emd2mi
                    best_val_loss = min(best_val_loss, val_losscd[i])
                    exppm = expansion_penalty.mean()
                    exppmi = exppm.item()
                    idx = random.randint(0, input.size()[0] - 1)
                    print(args.env + ' val [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f cd : %f'
                          %(epoch, i, len_val_dataset/args.batchSize, emd1mi,
                            emd2mi, exppmi, val_losscd[i]))
                    del(input)
                    del(output1)
                    del(output2)
            val_curve.append(np.mean(val_loss))
            val_curvecd.append(np.mean(val_losscd))

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            if val_losscd[i] < best_val_loss:
              best_val_loss = val_losscd[i]
              print('saving net...')
              torch.save(network.model.changed_state_dict(), '%s/network.pth' % (dir_name))

    logname = os.path.join(dir_name, 'log.txt')
    log_table = {
      "train_loss" : np.mean(train_loss),
      "val_loss" : np.mean(val_loss),
      "epoch" : epoch,
      "lr" : lrate,
      "bestval" : best_val_loss,

    }
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
  finally:
    del(dataset)
    del(dataloader)
    del(dataset_val)
    del(dataloader_val)
  return np.min(train_curve), np.min(val_curve), np.min(train_curvecd), np.min(val_curvecd)