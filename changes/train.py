import random
import torch.optim as optim
from transfered_model import *
from model import *
from utils import *
import os
import json
from dataset import *
from my_chamfer_interface import chamferDist
import gc
import time

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

def make_data_splits(args, trainp = './data/train.list',
    valp = './data/val.list',
    testp = './data/test.list'):
    perc_train = (1 - args.perc_val_data) * args.perc_data
    perc_val = args.perc_val_data * args.perc_data
    model_list = loadSplit('./data/all.list')
    tvsplitpos = int(perc_train * len(model_list))
    vtsplitpos = int((perc_train+perc_val) * len(model_list))
    random.shuffle(model_list)
    saveSplit(trainp, model_list[:tvsplitpos])
    saveSplit(valp, model_list[tvsplitpos:vtsplitpos])
    saveSplit(testp, model_list[vtsplitpos:])

def test(network, args, testp = './data/test.list'):
    dataset_test = ShapeNet(testp, npoints=args.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batchSize,
                                             shuffle=False, num_workers=args.workers, drop_last=True)
    cd, emd1mi, emd2mi, exppmi = validate(network,dataloader_test,100,None)
    print(args.env + ' test emd1: %f emd2: %f expansion_penalty: %f cd : %f'
          % (emd1mi, emd2mi, exppmi, cd))
    return cd, emd1mi, emd2mi, exppmi


def validate(network, dataloader, num_its_emd, iter_limit):
    network.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if iter_limit is not None and i > iter_limit:
                break
            id, input, gt = data
            input = input.float().cuda().transpose(1, 2)
            gt = gt.float().cuda()
            output1, output2, emd1, emd2, expansion_penalty = network(input, gt.contiguous(), 0.004, num_its_emd)
            emd1m = emd1.mean()
            emd1mi = emd1m.item()
            emd2m = emd2.mean()
            emd2mi = emd2m.item()
            dist1, dist2 = chamferDist()(output2.float(), gt)
            cd = (torch.mean(dist2) + torch.mean(dist1)).item()
            exppmi = expansion_penalty.mean().item()
            del (input)
            del (output1)
            del (output2)
    return cd, emd1mi, emd2mi, exppmi


def batchnum(epoch,batchind, loader):
    return len(loader)*epoch+batchind

def trainFull(network, dir_name, args, logevery = 100, lrate=0.001, kfacargs=defaultKFACargs,
              trainp='./data/train.list',
              valp='./data/val.list'):
    startt = time.process_time()
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
        optimizer = optim.Adam(network.parameters(), lr=lrate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
    train_curve = []
    val_curve = []
    train_curvecd = []
    val_curvecd = []
    val_batches = []
    labels_generated_points = (
        torch.Tensor(range(1, (args.n_primitives + 1) * (args.num_points // args.n_primitives) + 1))
        .view(args.num_points // args.n_primitives, (args.n_primitives + 1)).transpose(0, 1))
    labels_generated_points = (labels_generated_points) % (args.n_primitives + 1)
    labels_generated_points = labels_generated_points.contiguous().view(-1)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    best_val_loss = np.inf
    network.to("cuda")
    dataset = ShapeNet(trainp, npoints=args.num_points)
    dataset_val = ShapeNet(valp, npoints=args.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                             shuffle=True, num_workers=args.workers, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batchSize,
                                                 shuffle=False, num_workers=args.workers, drop_last=True)
    assert (len(dataset_val) // args.batchSize > 0), f"{len(dataset_val)} // {args.batchSize} == 0"
    len_dataset = len(dataset)
    len_val_dataset = len(dataset_val)
    print("Train Set Size: ", len_dataset)
    try:
        for epoch in range(args.nepoch):
            train_loss = np.zeros(logevery)
            train_losscd = np.zeros(logevery)
            # TRAIN MODE
            network.train()

            # learning rate schedule
            if usefirstorder:
                if epoch == 1:
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)
                if epoch == 2:
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

            for i, data in enumerate(dataloader, 0):
                batch_number = batchnum(epoch, i, dataloader)
                if usefirstorder:
                    lr_scheduler.step()
                else:
                    optimizer.lr = optimizer.lr * 0.98
                if args.epoch_iter_limit is not None and i > args.epoch_iter_limit:
                    break
                optimizer.zero_grad()
                id, input, gt = data
                input = input.float().cuda().transpose(1, 2)
                gt = gt.float().cuda()

                output1, output2, emd1, emd2, expansion_penalty = network(input, gt.contiguous(), 0.005, 10)
                emd1m = emd1.mean()
                emd1mi = emd1m.item()
                emd2m = emd2.mean()
                emd2mi = emd2m.item()
                exppm = expansion_penalty.mean()
                exppmi = exppm.item()
                loss_net = emd1m + emd2m + exppm * 0.1
                dist1, dist2 = chamferDist()(output2.float(), gt)
                cd = torch.mean(dist2) + torch.mean(dist1)
                if batch_number % logevery == 0:
                    print(args.env + ' train [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f cd : %f'
                          % (epoch, i, len_dataset / args.batchSize, emd1mi, emd2mi,
                             exppmi, cd))
                train_losscd[batch_number % logevery] = cd
                train_loss[batch_number % logevery] = emd2mi
                loss_net.backward()
                optimizer.step()

                del (input)
                del (output1)
                del (output2)
                if batch_number % logevery == 0:
                    cd, emd1mi, emd2mi, exppmi = validate(network, dataloader_val, 20, args.epoch_iter_limit_val)
                    best_val_loss = min(best_val_loss, cd)
                    print(args.env + ' val [%d: %d/%d]  emd1: %f emd2: %f expansion_penalty: %f cd : %f'
                          % (epoch, i, len_dataset / args.batchSize, emd1mi,
                             emd2mi, exppmi, cd))
                    print(f"mean train emd2 : {np.mean(train_loss)},cd {np.mean(train_losscd)}, time {time.process_time()-startt}")
                    train_curve.append(np.mean(train_loss))
                    train_curvecd.append(np.mean(train_losscd))
                    val_curve.append(np.mean(emd2mi))
                    val_curvecd.append(np.mean(cd))
                    curves = np.stack([np.array(train_curve), np.array(val_curve), np.array(train_curvecd), np.array(val_curvecd)])


                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, exist_ok=True)
                    np.save(f"{dir_name}/curves", curves)

                    if batch_number % logevery == 0:
                        print(f'saving net... this {emd2mi} best {best_val_loss}')
                        torch.save(network.model.state_dict(), f'{dir_name}/network{batch_number}.pth')

                    logname = os.path.join(dir_name, 'log.txt')
                    log_table = {
                        "train_loss": np.mean(train_loss),
                        "val_loss": cd,
                        "epoch": epoch,
                        "lr": lrate,
                        "bestval": best_val_loss,

                    }
                    with open(logname, 'a') as f:
                        f.write('json_stats: ' + json.dumps(log_table) + '\n')

    finally:
        del (dataset)
        del (dataloader)
        del (dataset_val)
        del (dataloader_val)
    return curves
