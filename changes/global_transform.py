from nn_utils import *
class GlobalTransformDepthSepLean(nn.Module):
    def __init__(self, partial_size, sizes, latents, globalv_size =None, translate_out_size = None):
        super().__init__()
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes)
        self.convs = nn.ModuleList([nn.Sequential(*make(sizes, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
                                    for i in range(partial_size // latents)])
        self.register_buffer('identity', torch.diag(torch.ones(self.latents)))
        self.translate = translate_out_size is not None
        self.use_globalv = globalv_size is not None
        if self.translate:
            self.translation = LinearBN(globalv_size, translate_out_size)

        self.fcs = nn.Sequential(*make([sizes[-1] + globalv_size if self.use_globalv else sizes[-1],
                                                       self.latents * self.latents],
                                                      lambda x, y: LinearBN(x, y)))
    def forward(self, partial, x, globalv):
        """
        Args:
            partial: The batched pointclouds of channel_size partial_size.
            globalv: Global feature of size globalvsize.

        Returns:
            X transformed with a transform generated from partial and globalv.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        bs = x.shape[0]
        x = x.view(bs, x.shape[1] // self.latents, self.latents, x.shape[2]).contiguous()
        outs = []
        for i in range(x.shape[1]):
            transform_select = self.convs[i](partial)
            softmaxweights = F.softmax(transform_select, 2)
            transform = (softmaxweights * transform_select).sum(2)
            if self.use_globalv:
                transform = self.fcs(torch.cat([transform, globalv], 1))
            else:
                transform = self.fcs(transform)
            transform = transform.view(-1, self.latents, self.latents)
            identity = torch.broadcast_to(self.identity.unsqueeze(0),
                                          (bs, self.identity.shape[0], self.identity.shape[1]))
            transform = transform + identity
            outs.append(torch.matmul(transform, x[:, i, :, :]))
        return torch.cat(outs, 1) + self.translation(globalv) if self.translate else torch.cat(outs, 1)
class GlobalTransformDepthSep(nn.Module):
    def __init__(self, partial_size, sizes, latents, globalv_size =None, translate_out_size = None):
        super().__init__()
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes) + (self.latents * self.latents,)
        self.convs = nn.ModuleList([nn.Sequential(*make(sizes, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
                                    for i in range(partial_size // latents)])
        self.register_buffer('identity', torch.diag(torch.ones(self.latents)))
        self.translate = translate_out_size is not None
        self.use_globalv = globalv_size is not None
        if self.translate:
            self.translation = LinearBN(globalv_size, translate_out_size)

        self.fcs = nn.Sequential(*make([self.latents * self.latents + globalv_size if self.use_globalv else self.latents * self.latents,
                                                       self.latents * self.latents],
                                                      lambda x, y: LinearBN(x, y)))

    def forward(self, partial, x, globalv):
        """
        Args:
            partial: The batched pointclouds of channel_size partial_size.
            globalv: Global feature of size globalvsize.

        Returns:
            X transformed with a transform generated from partial and globalv.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        bs = x.shape[0]
        x = x.view(bs, x.shape[1] // self.latents, self.latents, x.shape[2]).contiguous()
        outs = []
        for i in range(x.shape[1]):
            transform_select = self.convs[i](partial)
            softmaxweights = F.softmax(transform_select, 2)
            transform = (softmaxweights * transform_select).sum(2)
            if self.use_globalv:
                transform = self.fcs(torch.cat([transform, globalv], 1))
            else:
                transform = self.fcs(transform)
            transform = transform.view(-1, self.latents, self.latents)
            identity = torch.broadcast_to(self.identity.unsqueeze(0),
                                          (bs, self.identity.shape[0], self.identity.shape[1]))
            transform = transform + identity
            outs.append(torch.matmul(transform, x[:, i, :, :]))
        return torch.cat(outs, 1) + self.translation(globalv) if self.translate else torch.cat(outs, 1)


class AdditionalEncoder(nn.Module):
    def __init__(self, sizes, latents, out_size = 1024, depthseplean = False):
        super().__init__()
        sizes1 = (3 + 128,) + tuple(sizes[:-1])
        bottleneck_size = sizes[-1]
        sizes2 = tuple(sizes[-2:])
        self.extractor = nn.Identity() if True else BatchNormConv1D(bottleneck_size, bottleneck_size)
        self.transform_extractor = LinearBN(bottleneck_size, out_size)
        self.convs1 = nn.Sequential(*make(sizes1, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
        self.convs2 = nn.Sequential(*make(sizes2, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
        self.gt = (GlobalTransformDepthSepLean(sizes[-2], sizes[-2], latents, out_size) if depthseplean else
                   GlobalTransformDepthSep(sizes[-2], (3 + 128,), latents, out_size))


    def forward(self, partial3, partial128, x, xfc):
        partial = torch.cat([partial3, partial128], 1)
        v = self.convs1(partial)
        v = self.gt(v, v, x)
        v = self.convs2(v)
        softmaxweights = F.softmax(v, 2)
        v = 0.05*self.transform_extractor((softmaxweights * self.extractor(v)).sum(2))
        return v + xfc


class AdditionalEncoderIndentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, partial3, partial128, x, xfc):
        return x


class GlobalTransformIndentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, partial, x):
        return x


class GlobalTransformIndentityGlobalV(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, partial, x, globalv):
        return x
