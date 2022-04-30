import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from dltools.callback import TrainEvalCallback
from dltools.exceptions import CancelBatchException, CancelEpochException, CancelTrainException
from dltools.functions import flatten, listify, init_cnn  # , init_cnn init_cnn is loaded in __init__
from dltools.layer import Lambda, GeneralRelu, lin_bn_layer


class Learner():
    def __init__(self, model, opt, loss_func, data, cbs=None, cb_funcs=None):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.data = data
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop = False
        self.cbs = [TrainEvalCallback()] + cbs

    def one_batch(self, xb, yb):
        try:
            self.xb, self.yb = xb, yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train:
                return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:
            self('after_cancel_batch')
        finally:
            self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in dl:
                self.one_batch(xb, yb)
        except CancelEpochException:
            self('after_cancel_epoch')            

    def fit(self, epochs):
        """
        Trains for n epochs.
        Using data, optimizer, loss_func and model.
        """
        self.epochs = epochs
        # self.learn = learn
        self.loss = tensor(0.)
        try:
            for cb in self.cbs:
                # cb.set_runner(self)
                cb.set_learner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'):
                    self.all_batches(self.data.train_dl)
                with torch.no_grad():
                    if not self('begin_validate'):
                        self.all_batches(self.data.valid_dl)
                self('after_epoch')
        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')
            # self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res


# class Learner():
#     """
#     Container class for model, optimizer, loss function and data
#     """
#     def __init__(self, model, opt, loss_func, data):
#         self.model = model
#         self.opt = opt
#         self.loss_func = loss_func
#         self.data = data


def get_cnn_grelu_layers(data, nfs, leak, clamp, layer, **kwargs):
    nfs = [1] + nfs
    leak = [0. for _ in range(len(nfs))] if leak is None else leak
    clamp = [0. for _ in range(len(nfs))] if clamp is None else clamp
    return [layer(nfs[i], nfs[i+1], 5 if i == 0 else 3, leak=leak[i], sub=clamp[i], **kwargs) for i in range(len(nfs)-1)] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c)]


def get_cnn_grelu_model(data, nfs, leak, clamp, layer, **kwargs):
    return nn.Sequential(*get_cnn_grelu_layers(data, nfs, leak, clamp, layer, **kwargs))


def get_cnn_layers(data, nfs, layer, **kwargs):
    nfs = [1] + nfs
    return [layer(nfs[i], nfs[i+1], 5 if i == 0 else 3, **kwargs) for i in range(len(nfs)-1)] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c)]


def get_cnn_model(data, nfs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))


# def get_lin_layers(data, nh, layer, **kwargs):
#     """
#     Returns nn.Sequential for each element in nh based on layer. The activation sizes are based on nh.
#      - nh: num hidden, i.e. a list of activation sizes. Input size is given by num of rows of the prev layer.
#      - layer: single layer, e.g. nn.Linear, or set of layers, e.g. [nn.Linear, nn.ReLU]
#     """
#     nh = [data.train_ds.tensors[0].shape[1]] + nh + [data.c]
#     return [layer(nh[i], nh[i+1]) for i in range(len(nh)-1)]


def get_lin_layers(data, nh, layer, **kwargs):
    """
    Returns nn.Sequential for each element in nh based on layer. The activation sizes are based on nh.
     - nh: num hidden, i.e. a list of activation sizes. Input size is given by num of rows of the prev layer.
     - layer: single layer, e.g. nn.Linear, or set of layers, e.g. [nn.Linear, nn.ReLU]
    """
    nh = [data.train_ds.tensors[0].shape[1]] + nh
    return [layer(nh[i], nh[i+1]) for i in range(len(nh)-1)] + [nn.Linear(nh[-1], data.c)]


def get_lin_grelu_layers(data, nh, leak, sub, maxv, layer, bn, **kwargs):
    if not isinstance(leak, list):
        leak = [leak] * len(nh)
    if not isinstance(sub, list):
        sub = [sub] * len(nh)
    if not isinstance(maxv, list):
        maxv = [maxv] * len(nh)
    assert len(nh) == len(leak) == len(sub) == len(maxv), 'Leak, sub or maxv has to be a single float or a list of floats with the same amount of values as there are layers.'
    nh = [data.train_ds.tensors[0].shape[1]] + nh
    return [layer(nh[i], nh[i+1], nh[i+1], bn, leak=leak[i], sub=sub[i], maxv=maxv[i]) for i in range(len(nh)-1)] + [nn.Linear(nh[-1], data.c), GeneralRelu(0., 0., None)]


def get_lin_grelu_layers_2(data, nh, layer, bn, leak, sub, maxv, **kwargs):    
    if not isinstance(leak, list):        
        leak = [leak] * len(nh)
    if not isinstance(sub, list):
        sub = [sub] * len(nh)
    if not isinstance(maxv, list):
        maxv = [maxv] * len(nh)
    assert len(nh) == len(leak) == len(sub) == len(maxv), 'Leak, sub or maxv has to be a single float or a list of floats with the same amount of values as there are layers.'    
    nh = [data.train_ds.tensors[0].shape[1]] + nh
    return [layer(nh[i], nh[i+1], leak=leak[i], sub=sub[i], maxv=maxv[i]) for i in range(len(nh)-1)] + [layer(nh[-1], data.c, leak=0., sub=0., maxv=None)]


def get_lin_grelu_model(data, nh, leak, sub, maxv, layer, bn=True, **kwargs):
    return nn.Sequential(*get_lin_grelu_layers(data, nh, leak, sub, maxv, layer, bn, **kwargs))


def get_lin_grelu_model_2(data, nh, layer, bn=False, **kwargs):
    return nn.Sequential(*get_lin_grelu_layers_2(data, nh, layer, bn, **kwargs))


def get_lin_model(data, nh, layer, **kwargs):
    return nn.Sequential(*get_lin_layers(data, nh, layer, **kwargs))


# def get_runner(model, data, lr=0.6, cbs=None, opt_func=None, loss_func=F.cross_entropy):
#     if opt_func is None:
#         opt_func = optim.SGD
#     opt = opt_func(model.parameters(), lr=lr)
#     learn = Learner(model, opt, loss_func, data)
#     return learn, Runner(cb_funcs=listify(cbs))


# def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, uniform=False, **kwargs):
#     model = get_cnn_model(data, nfs, layer, **kwargs)
#     init_cnn(model, uniform=uniform)
#     return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)


# def get_lin_learn_run(data, nh, lr, layer, cbs=None, opt_func=None, loss_func=None, uniform=False, **kwargs):
#     model = get_lin_grelu_model_2(data, nh, layer=layer, bn=False, **kwargs)
#     init_cnn(model, uniform=uniform)
#     learn, run = get_runner(model=model, data=data, lr=lr, cbs=cbs, opt_func=opt_func, loss_func=loss_func)
#     return model, learn, run


def get_lin_learner(data, nh, lr, layer, cbs=None, opt_func=None, loss_func=None, uniform=False, **kwargs):
    model = get_lin_grelu_model_2(data, nh, layer=layer, bn=False, **kwargs)
    init_cnn(model, uniform=uniform)
    if opt_func is None:
        opt_func = optim.SGD
    opt = opt_func(model.parameters(), lr=lr)
    return Learner(model, opt, loss_func, data, cb_funcs=listify(cbs))
