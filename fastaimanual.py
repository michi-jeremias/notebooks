# nb 00

import operator

def test(a,b, cmp, cname=None):
    if cname is None:
        cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"


def test_eq(a,b):
    test(a,b,operator.eq,'==')

""" nb01 """
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'


def near(a,b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)


def test_near(a,b):
    test(a, b, near)


""" nb02 fully connected """
from torch.nn import init
# import torch.nn


def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path,'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train, y_train, x_valid, y_valid))


def normalize(x, m, s):
    return (x-m)/s


def test_near_zero(a, tol=1e-3):
    assert a.abs() < tol, f"Near zero: {a}"


def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()

from IPython.core.debugger import set_trace
from torch import nn
import torch.nn.functional as F


"""NB03"""


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()

from torch import optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


class Dataset():
    def __init__(self, x, y):
        self.x, self.y = x, y
        assert len(self.x) == len(self.y), "Tensors have different length"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return(
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs*2, shuffle=False, **kwargs)
        )


""" NB04 """
import fastprogress

def get_model(data, lr=0.5, nh=50):
    m = data.train_ds.x.shape[1]
    model = nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, data.c))
    return model, optim.SGD(model.parameters(), lr)


class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c
   
    @property
    def train_ds(self):
        return self.train_dl.dataset
        
    @property
    def valid_ds(self):
        return self.train_ds.dataset


import re
from typing import Iterable
from functools import partial


def camel2snake(name):
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


class Callback():
    _order = 0
    def set_runnner(self, run):
        self.run = run
    
    def __getattr__(self, k):
        return getattr(self.run, k)
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    

class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0
        self.run.n_iter = 0
    
    def after_batch(self):
        if not self.in_train: return        
        self.run.n_epochs += 1./self.iters
        self.run.n_iter += 1
    
    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True
    
    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop = False
        self.cbs = [TrainEvalCallback()] + cbs
    
    @property
    def opt(self): return self.learn.opt

    @property
    def model(self): return self.learn.model

    @property
    def loss_func(self): return self.learn.loss_func

    @property
    def data(self): return self.learn.data

    def one_batch(self, xb, yb):
        self.xb, self.yb = xb, yb
        if self('begin_batch'): return
        self.pred = self.model(self.xb)
        if self('after_pred'): return
        self.loss = self.loss_func(self.pred, self.yb)
        if self('after_loss') or not self.in_train: return
        self.loss.backward()
        if self('after_backward'): return
        self.opt.step()
        if self('after_step'): return
        self.opt.zero_grad()
    
    def all_batches(self, dl):
        self.iters = len(dl)
        for xb, yb in dl:
            if self.stop: break
            self.one_batch(xb, yb)
            self('after_batch')
        self.stop = False
    
    def fit(self, epochs, learn):
        self.epochs = epochs
        self.learn = learn

        try:
            for cb in self.cbs:
                cb.set_runnner(self)
            if self('begin_fit'): return
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'):
                    self.all_batches(self.data.train_dl)
                
                with torch.no_grad():
                    if not self('begin_validate'):
                        self.all_batches(self.data.valid_dl)
                if self('after_epoch'): break
        finally:
            self('after_fit')
            self.learn = None
        
    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False
    

class AvgStats():
    def __init__(self, metrics, in_train):
        self.metrics = listify(metrics)
        self.in_train = in_train
    
    def reset(self):
        self.tot_loss = 0.
        self.count = 0
        self.tot_mets = [0.] * len(self.metrics)
    
    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets
    
    @property
    def avg_stats(self):
        return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
    
    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn
    

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
    
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)
    
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.data = data

""" NB05 """
def create_learner(model_func, loss_func, data):
    return Learner(*model_func(data), loss_func, data)
