from nn_utils import *


class GlobalTransformDepthSep(nn.Module):
    def __init__(self, partial_size, globalvsize, sizes, latents, use_globalv=True):
        super().__init__()
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes) + (self.latents * self.latents,)
        self.convs = nn.ModuleList([nn.Sequential(*make(sizes, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
                                    for i in range(partial_size // latents)])
        self.register_buffer('identity', torch.diag(torch.ones(self.latents)))
        self.use_globalv = use_globalv
        if self.use_globalv:
            self.fcs = nn.Sequential(*make([self.latents * self.latents + globalvsize,
                                                           self.latents * self.latents],
                                                          lambda x, y: LinearBNRelu(x, y)))

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
            transform = transform.view(-1, self.latents, self.latents)
            identity = torch.broadcast_to(self.identity.unsqueeze(0),
                                          (bs, self.identity.shape[0], self.identity.shape[1]))
            transform = transform + identity
            outs.append(torch.matmul(transform, x[:, i, :, :]))
        return torch.cat(outs, 1)


class GlobalAdditiveGeneral(nn.Module):
    def __init__(self, bottleneck_size, partial_size, globalvsize, sizes, latents):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.latents = latents
        sizes = (partial_size,) + tuple(sizes) + (bottleneck_size,)
        self.convs = nn.Sequential(*make(sizes, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
        self.usefcs = True
        if self.usefcs:
            self.fcs = nn.Sequential(*make([globalvsize, bottleneck_size * bottleneck_size, bottleneck_size],
                                           lambda x, y: LinearBNRelu(x, y)))

    def forward(self, x, globalv):
        v = self.convs(x)
        softmaxweights = F.softmax(v, 2)
        v = (softmaxweights * v).sum(2)
        if self.usefcs:
            v = v + self.fcs(globalv)
        os = x.shape
        if len(x.shape) < 2:
            x = x.unsqueeze(2)
        return (v.unsqueeze(2).broadcast_to(x.shape) + x).reshape(os)


class AdditionalEncoder(nn.Module):
    def __init__(self, sizes, latents, bottleneck_size=1024):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        sizes1 = (3 + 128,) + tuple(sizes[:-1])
        sizes2 = tuple(sizes[-2:]) + (bottleneck_size,)
        self.transform_extractor = BatchNormConv1DNoAct(bottleneck_size, bottleneck_size)
        self.convs1 = nn.Sequential(*make(sizes1, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
        self.convs2 = nn.Sequential(*make(sizes2, lambda x, y: nn.Sequential(BatchNormConv1D(x, y))))
        self.gt = GlobalTransformDepthSep(sizes[-2], bottleneck_size, (3 + 128,), latents)

    def forward(self, partial3, partial128, x, xfc):
        partial = torch.cat([partial3, partial128], 1)
        v = self.convs1(partial)
        v = self.gt(v, v, x)
        v = self.convs2(v)
        softmaxweights = F.softmax(v, 2)
        v = (softmaxweights * self.transform_extractor(v)).sum(2)
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
