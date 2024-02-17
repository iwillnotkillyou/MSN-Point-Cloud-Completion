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
import trimesh
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

def make_data_splits(args, allp, trainp = './data/train.list',
    valp = './data/val.list',
    testp = './data/test.list'):
    perc_train = (1 - args.perc_val_data) * args.perc_data
    perc_val = args.perc_val_data * args.perc_data
    model_list = loadSplit(allp)
    tvsplitpos = int(perc_train * len(model_list))
    vtsplitpos = int((perc_train+perc_val) * len(model_list))
    random.shuffle(model_list)
    saveSplit(trainp, model_list[:tvsplitpos])
    saveSplit(valp, model_list[tvsplitpos:vtsplitpos])
    saveSplit(testp, model_list[vtsplitpos:])
def exportf(tensor,name):
    ps = [tensor[0, i, :].detach().cpu().numpy() for i in range(tensor.shape[1])]
    ls = [f"v {p[0]} {p[1]} {p[2]}\n" for p in ps]
    open(f"{name}.obj","w").writelines(ls)
def printf(inp, output2, target, i, name, fol):
    os.makedirs(fol,exist_ok= True)
    exportf(inp.transpose(1,2), f"{fol}/{i}inp")
    #exportf(output1, f"{fol}/{i}out1{name}")
    exportf(output2, f"{fol}/{i}out2{name}")
    exportf(target, f"{fol}/{i}targ{name}")

def printf2(inp, output2, target, i, name, fol):
    if i % 10 == 0 and (i//50) & 100 == 0:
        printf(inp,output2,target,i,name,fol)
def test(network, dataloader_test, name, fol):


    vs = validate(network,dataloader_test,100,None, lambda x,y,z,i :printf2(x,y,z,i,name,fol))
    cd, emd1mi, emd2mi, exppmi = vs
    cdm, emd1mim, emd2mim, exppmim = (np.mean(v) for v in vs)
    print('test emd1: %f emd2: %f expansion_penalty: %f cd : %f'
          % (emd1mim, emd2mim, exppmim, cdm))
    return cd, emd1mi, emd2mi, exppmi


def validate(network, dataloader, num_its_emd, iter_limit, printf = None):
    network.eval()
    cd = []
    emd1mi = []
    emd2mi = []
    exppmi = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if iter_limit is not None and i > iter_limit:
                break
            id, inp, gt = data
            inp = inp.float().cuda().transpose(1, 2)
            gt = gt.float().cuda()
            output1, output2, emd1, emd2, expansion_penalty = network(inp, gt.contiguous(), 0.004, num_its_emd)
            if printf is not None:
                printf(inp, output1, output2, i)
            emd1m = emd1.mean()
            emd1mi.append(emd1m.item())
            emd2m = emd2.mean()
            emd2mi.append(emd2m.item())
            dist1, dist2 = chamferDist()(output2.float(), gt)
            cd.append((torch.mean(dist2) + torch.mean(dist1)).item())
            exppmi.append(expansion_penalty.mean().item())
            del (inp)
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
                    vs = validate(network, dataloader_val, 20, args.epoch_iter_limit_val)
                    cd, emd1mi, emd2mi, exppmi = (np.mean(v) for v in vs)
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

                    if batch_number % logevery*2 == 0:
                        print(f'saving net... this {cd} best {best_val_loss}')
                        torch.save(network.model.state_dict(), f'{dir_name}/network{batch_number}.pth')
                        torch.save(network.model.state_dict(), f'{dir_name}/network.pth')

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

if __name__ == "__main__":
    from full_model import *
    from transfered_model import *
    import torch
    from test_on_completion3d import test_on_completion3D
    from matplotlib import pyplot as plt
    class completionargs:
      def __init__(self,model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.nworkers = 2
        self.dataset = 'shapenet'
        self.pc_augm_scale=0
        self.pc_augm_rot=0
        self.pc_augm_mirror_prob=0
        self.pc_augm_jitter=0
        self.inpts=500
      def step(self, targets, clouds_data):
        clouds_data = torch.from_numpy(clouds_data[1]).cuda()
        targets = torch.from_numpy(targets).cuda()
        output1, output2, emd1, emd2, expansion_penalty = self.model(clouds_data, targets.contiguous(), 0.005, 10)
        dist1, dist2 = chamferDist()(output2.float(),targets.float())
        loss = torch.mean(dist2) + torch.mean(dist1)
        return loss.item(), dist1, dist2, emd2, output2.detach().cpu().numpy()


    def make_fuller_model(name, args, architect_args):
        def additionalencoderf():
            if architect_args.additional_sizes is None:
                return AdditionalEncoderIndentity()
            return AdditionalEncoder(architect_args.additional_sizes, architect_args.additional_latents)

        def pointnetfeatf():
            if architect_args.modif_sizes is None:
                return PointNetfeatFreeze()
            if architect_args.modif_partial:
                return PointNetfeatReturn2TPartial(architect_args.modif_sizes)
            return PointNetfeatReturn2(architect_args.modif_sizes, architect_args.modif_latents)

        def additionaldecoderf():
            return GlobalTransformDepthSep(3, architect_args.additional_dec_sizes, 3, 1024)

        additionaldecoderf = None if architect_args.additional_dec_sizes is None else additionaldecoderf
        model = TransformMSN(additionalencoderf, pointnetfeatf, additionaldecoderf, args.num_points,
                             architect_args.train_encoder)

        network = FullModel(model)
        # network = torch.nn.DataParallel(network)
        # model.apply(weights_init) #initialization of the weight
        if args.base_model != '':
            previous_state_dict = torch.load(args.base_model)
            model.load_state_dict(previous_state_dict, False, True)
            print("Base weight loaded ")
        elif name != '':
            previous_state_dict = torch.load(name)
            model.load_state_dict(previous_state_dict, False, True)
            print("Previous weight loaded ")
        model.freeze()
        return network


    def make_base_model(args):
        model = MSN()
        network = FullModel(model)
        if args.base_model != '':
            previous_state_dict = torch.load(args.base_model)
            model.load_state_dict(previous_state_dict, False, True)
            print("Base weight loaded ")
        return network


    train = True


    class opt:
        def __init__(self):
            self.base_model = '/content/repo_folder/MSN-Point-Cloud-Completion/trained_model/network.pth'
            self.epoch_iter_limit_val = 10
            self.batchSize = 16
            self.epoch_iter_limit = None
            self.perc_data = 0.85
            self.perc_val_data = 0.1
            self.workers = 1
            self.nepoch = 3
            self.model = ''
            self.num_points = 8192
            self.n_primitives = 16
            self.env = "MSN_TRAIN"
            self.manualSeed = 0
            self.run_embedder = True
            self.embeddingpath = "/content/embeddings"
            self.usefirstorder = True


    class architect_opts:
        def __init__(self, additional_enc=None, modif=None, additional_dec=None, shared_latents=None,
                     modif_partial=True, train_encoder=False, train=train):
            dontuseshared = shared_latents is None

            def getval(tup):
                return tup[1] if dontuseshared and tup is not None else tup, tup[
                    0] if dontuseshared and tup is not None else shared_latents

            self.additional_sizes, self.additional_latents = getval(additional_enc)
            self.modif_sizes, self.modif_latents = getval(modif)
            self.additional_dec_sizes, self.additional_dec_latents = getval(additional_dec)
            self.train = train
            self.modif_partial = modif_partial
            self.train_encoder = train_encoder


    arcargsd = {}

    argsd = {}
    arcargsd = {}
    arcargsd["m1"] = architect_opts((32, (128, 128)), None, None, train=False)
    arcargsd["m2"] = architect_opts((32, (256, 256)), None, None, train=False, train_encoder=True)
    # arcargsd["m2"] = architect_opts((16,(64,64)),None,train = True)
    # arcargsd["m3"] = architect_opts((32,(128,128)),(32,(64,)),train = True)
    # arcargsd["m4"] = architect_opts((128,128),(64,),32,32,train = True)
    # arcargsd["m5"] = architect_opts((128,128),(128,),32,16,modif_partial = False, train = True)
    # arcargsd["m6"] = architect_opts((256,256),(64,),32,32, train = True)
    # arcargsd["m7"] = architect_opts((128,128),(256,),32,32,modif_partial = False, train = True)
    # arcargsd["m8"] = architect_opts((128,128),None,16,16,train = False)

    torch.cuda.empty_cache()
    names = arcargsd.keys()

    for name in names:
        argsd[name] = opt()


    def makeso(args):
        args.usefirstorder = False
        args.batchSize = 64


    drivesplitf = "/content/drive/MyDrive/saved"
    # makeso(argsd["m2"])
    # makeso(argsd["m4"])
    # makeso(argsd["m6"])
    resplit = True
    trainp = f'{drivesplitf}/train.list'
    valp = f'{drivesplitf}/val.list'
    testp = f'{drivesplitf}/test.list'
    if resplit:
        make_data_splits(opt(), f'{drivesplitf}/all.list',
                         trainp,
                         valp,
                         testp)


    def plotrainval(save_path):
        p = f"{save_path}/curves.npy"
        if not os.path.isfile(p):
            return
        curves = np.load(p)
        plt.plot(curves[0, :], label="train emd")
        plt.plot(curves[1, :], label="val emd")
        plt.legend()
        plt.show()
        plt.plot(curves[2, :], label="train cd")
        plt.plot(curves[3, :], label="val cd")
        plt.legend()
        plt.show()


    if train:
        for name in names:
            if arcargsd[name].train:
                save_path = f"/content/drive/MyDrive/saved/{name}"
                model = make_fuller_model("", argsd[name], arcargsd[name])
                cs = 0
                for c in model.model.children():
                    csc = 0
                    for param in c.parameters():
                        if param.requires_grad:
                            csc += param.nelement()
                    cs += csc
                    print(type(c), csc, cs)
                trainFull(model, save_path, argsd[name], 200, 1e-3, trainp=trainp,
                          valp=valp)
                plotrainval(save_path)

    base_model = make_base_model(opt())


    def print_results_sorted(results):
        vs = [(x, results[x]) for x in results]
        print([vs[x] for x in np.argsort([x[1] for x in vs])])


    dir = "/content/repo_folder/completion3d/data"
    link = "http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip"
    dataf = "/content/data.zip"
    testfunc = lambda x: test_on_completion3D(completionargs(x, 4), False)

    dataset_test = ShapeNet(testp, npoints=args.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                  shuffle=False, num_workers=args.workers, drop_last=True)
    resultsavef = f'{drivesplitf}/output'
    results = {}
    if True:
        model = make_base_model(opt())
        model.cuda()
        results["base_model"] = testfunc(model)
        basemodelvs = test(model, dataloader_test, "base", resultsavef)
        print_results_sorted(results)
        del (model)

    for name in list(names):
        print(name)
        save_path = f"/content/drive/MyDrive/saved/{name}"
        save_file = f'{save_path}/network.pth'
        print(os.path.isfile(save_file))
        if not os.path.isfile(save_file):
            continue
        plotrainval(save_path)
        model = make_fuller_model(save_file, argsd[name], arcargsd[name])
        modelvs = test(model, dataloader_test, name, resultsavef)
        modelvs[0]
        model.cuda()
        results[name] = testfunc(model)
        print_results_sorted(results)
        del (model)