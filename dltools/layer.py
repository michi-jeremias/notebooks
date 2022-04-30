import torch.nn.functional as F
from torch import nn


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.maxv = maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x

    def __repr__(self):
        res = f"{self.__class__.__name__} (leak={self.leak}, sub={self.sub}, maxv={self.maxv})"
        return res


class Lambda(nn.Module):
    """
    Class that akes a basic function and converts it to a layer.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class LinLayer(nn.Module):
    def __init__(self, feat_in, feat_out, leak=0.0, sub=0.0, maxv=None, **kwargs):
        super().__init__()
        self.lin = nn.Linear(feat_in, feat_out, bias=True)
        self.relu = GeneralRelu(leak, sub, maxv)

    def forward(self, x):
        return self.relu(self.lin(x))

    @property
    def bias(self):
        return -self.relu.sub

    @bias.setter
    def bias(self, v):
        self.relu.sub = -v

    @property
    def weight(self):
        return self.lin.weight


def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride, bias=not bn),
        GeneralRelu(**kwargs),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


def lin_bn_layer(num_in, num_act, nh, bn=True, **kwargs):
    layers = [nn.Linear(num_in, num_act, bias=not bn), GeneralRelu(**kwargs)]
    if bn:
        layers.append(nn.BatchNorm1d(nh, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


def lin_layer(num_in, num_act, **kwargs):
    return nn.Sequential(nn.Linear(num_in, num_act), GeneralRelu(**kwargs))


def flatten(x):
    return x.view(x.shape[0], -1)
