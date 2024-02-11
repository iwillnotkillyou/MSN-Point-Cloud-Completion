import torch

def make_fuller_model(name, args, architect_args):
  def additionalencoderf(x,y):
    if not architect_args.use_gt:
      return AdditionalEncoderIndentity()
    return AdditionalEncoder(architect_args.additional_sizes,architect_args.latents)
  def encoder_modiff(x,y,z):
    if not architect_args.use_gt:
      return GlobalTransformIndentity()
    return GlobalTransformDepthSep(128,1024,architect_args.modif_sizes,architect_args.latents)
  model = TransformMSN(additional_encoderf = additionalencoderf, encoder_modiff= encoder_modiff)
  network = FullModel(model)
  #network = torch.nn.DataParallel(network)
  #model.apply(weights_init) #initialization of the weight

def make_base_model(args):
  model = MSN()
  network = FullModel(model)
  if args.base_model != '':
    previous_state_dict = torch.load(args.base_model)
    model.load_state_dict(previous_state_dict,False,True)
    print("Previous weight loaded ")
  return network


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

! git sync
train = True
class opt:
  def __init__(self):
    self.base_model = '/content/repo_folder/MSN-Point-Cloud-Completion/trained_model/network.pth'
    self.epoch_iter_limit_val = 10
    self.batchSize = 100
    self.epoch_iter_limit = None
    self.perc_data = 1 if train else 1
    self.perc_val_data = 0.1 if train else 0.999
    self.workers = 1
    self.nepoch = 10
    self.model = ''
    self.num_points = 8192
    self.n_primitives = 16
    self.env = "MSN_TRAIN"
    self.manualSeed = 0
    self.run_embedder = True
    self.embeddingpath = "/content/embeddings"
    self.usefirstorder = True

class architect_opts:
  def __init__(self, res_layer_sizes, gt_layer_sizes = None, gt_layer_sizesdec = None, latent_size = 64, latent_size_decoder = 3, gt_additive = False, gt_decadditive = False, train = train):
    self.use_gt = gt_layer_sizes != None
    self.use_gt_dec = gt_layer_sizesdec != None
    self.gt_layer_sizes = gt_layer_sizes
    self.gt_layer_sizesdec = gt_layer_sizesdec
    self.gt_decadditive = gt_decadditive
    self.latents = latent_size
    self.gt_additive = gt_additive
    self.latentsdec = latent_size_decoder
    self.res_layer_sizes = res_layer_sizes
    self.train = train

arcargsd = {}

argsd = {}
arcargsd = {}
#arcargsd["m1"] = architect_opts((512,256,128,64),train = False)
#arcargsd["m7"] = architect_opts((512,256,128,64),(256,), None, 32, gt_additive = True,train = False)
#arcargsd["m8"] = architect_opts((512,256,128,64),(256,),(256,),32,train = False)
#arcargsd["m10"] = architect_opts((512,256,128,64),(256,),None,32,train = False)
#arcargsd["m11"] = architect_opts((512,256,128,64),(256,), (256,), 32, gt_additive = True,train = False)
#arcargsd["m12"] = architect_opts((512,256,128,64),(256,), (256,), 64, gt_additive = True,  gt_decadditive = True,train = False)
#arcargsd["m13"] = architect_opts((512,256,128,64),(32,256,), (128,64,), 32, gt_additive = True,train = False)
#arcargsd["m14"] = architect_opts((512,256,128,64),(32,256,512), (128,64,), 32, gt_additive = True,train = False)
arcargsd["m15"] = architect_opts(None,None, None, 32, gt_additive = True,train = False)
arcargsd["m16"] = architect_opts(None,(32,64,256,512), (512,256,64,32), 32, gt_additive = True,train = False)
arcargsd["m17"] = architect_opts(None,(64,512,), (512,64,), 32, gt_additive = True,train = False)
arcargsd["m18"] = architect_opts(None, None, (256,), 32, gt_additive = True,train = False)

torch.cuda.empty_cache()
names = arcargsd.keys()

for name in names:
  argsd[name] = opt()

argsd["m18"].usefirstorder = False

%cd /content/repo_folder/MSN-Point-Cloud-Completion/
if train:
  for name in names:
    if arcargsd[name].train:
      save_path = f"/content/drive/MyDrive/saved/{name}"
      model = fullmodel.make_fuller_model("", argsd[name], arcargsd[name])
      cs = 0
      for c in model.model.children():
        csc = 0
        for param in c.parameters():
          if param.requires_grad:
            csc += param.nelement()
        cs += csc
        print(type(c), csc, cs)
      train.trainFull(model,save_path, False, argsd[name], 1e-3)

base_model = make_base_model(opt())
def print_results_sorted(results):
  vs = [(x,results[x]) for x in results]
  print([vs[x] for x in np.argsort([x[1] for x in vs])])
if first:
  dir = "/content/repo_folder/completion3d/data"
  link = "http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip"
  dataf = "/content/data.zip"
  ! wget -nc -O $dataf $link
  ! rm -r $dir
  ! mkdir -p $dir
  ! unzip -qq -o $dataf -d $dir
%cd /content/repo_folder/completion3d
testfunc = lambda x : test_on_completion3D(completionargs(x,4),False)

results = {}
if False:
  model = make_base_model(opt())
  model.cuda()
  results["base_model"] = testfunc(model)
  print_results_sorted(results)

for name in list(names):
  save_path = f"/content/drive/MyDrive/saved/{name}"
  save_file = f'{save_path}/network.pth'
  if not os.path.isfile(save_file):
    continue
  model = make_fuller_model(save_file, argsd[name], arcargsd[name])
  model.cuda()
  results[name] = testfunc(model)
  print_results_sorted(results)
  del(model)