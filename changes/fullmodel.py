from mychamfer import *
from transfered_model import *
class FullModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, gt, eps, iters):
        output1, output2, expansion_penalty = self.model(inputs)
        gt = gt[:, :, :3]
        idx = np.random.randint(0,gt.shape[1], self.model.num_points)
        gt = gt[:, idx, :]
        dist, _ = self.EMD(output1, gt, eps, iters)
        emd1 = torch.sqrt(dist).mean(1)

        dist, _ = self.EMD(output2, gt, eps, iters)
        emd2 = torch.sqrt(dist).mean(1)

        return output1, output2, emd1, emd2, expansion_penalty

class PointNetResFT(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = (256,128,64)
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
        self.convs = nn.Sequential(*make(sizes, lambda x,y : BatchNormConv1D(x,y)))
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
        self.convs = nn.Sequential(*make(sizes, lambda x,y : BatchNormConv1D(x,y)))
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
        softmaxweights = F.softmax(x,2)
        x = (softmaxweights*x).sum(2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = self.convs(x)
        x = self.th(self.lastconv(x))
        return x

def make_fuller_model(name, args, architect_args):
  def encFunc(x,y):
    if not architect_args.use_gt:
      return GlobalTransformIndentity()
    if architect_args.gt_additive:
      return GlobalTransformAdditive(x,y,architect_args.gt_layer_sizes,architect_args.latents)
    else:
      return GlobalTransform(x,y,architect_args.gt_layer_sizes,architect_args.latents)
  def decFunc(x,y,z):
    if not architect_args.use_gt_dec:
      return GlobalTransformIndentityGlobalV()
    if architect_args.gt_decadditive:
      return GlobalTransformAdditiveGeneral(x,y,z,architect_args.gt_layer_sizes,architect_args.latentsdec)
    else:
      return GlobalTransformGeneral(x,y,z,architect_args.gt_layer_sizes,architect_args.latentsdec)
  def residualFunc():
    return PointNetResFT() if architect_args.res_layer_sizes == None else PointNetResSoftMax(architect_args.res_layer_sizes)
  model = TransformMSN(residual = residualFunc, additional_encoder = encFunc, additional_decoder= decFunc)
  network = FullModel(model)
  #network = torch.nn.DataParallel(network)
  #model.apply(weights_init) #initialization of the weight

  if name != '':
      previous_state_dict = torch.load(name)
      network.load_state_dict(previous_state_dict,True,True)
      print("Previous weight loaded ")
  elif args.base_model != '':
      previous_state_dict = torch.load(args.base_model)
      model.load_state_dict(previous_state_dict,False,True)
      print("Previous weight loaded ")
  model.freeze()
  return network