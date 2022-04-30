import re
from dltools.functions import camel2snake, listify
from dltools.exceptions import CancelTrainException
from functools import partial
import torch
from matplotlib import pyplot as plt
import math


class AvgStats:
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0.0, 0
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


# # Version with runner
# class Callback():
#     _order = 0

#     def set_runner(self, run):
#         self.run = run

#     def __getattr__(self, k):
#         return getattr(self.run, k)

#     @property
#     def name(self):
#         name = re.sub(r'Callback$', '', self.__class__.__name__)
#         return camel2snake(name or 'callback')

#     def __call__(self, cb_name):
#         f = getattr(self, cb_name, None)
#         if f and f():
#             return True
#         return False


class Callback:
    _order = 0

    def set_learner(self, learn):
        self.learn = learn

    def __getattr__(self, k):
        return getattr(self.learn, k)

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False


# # Version with runner
# class AvgStatsCallback(Callback):
#     def __init__(self, metrics):
#         self.train_stats = AvgStats(metrics, True)
#         self.valid_stats = AvgStats(metrics, False)

#     def begin_epoch(self):
#         self.train_stats.reset()
#         self.valid_stats.reset()

#     def after_loss(self):
#         stats = self.train_stats if self.in_train else self.valid_stats
#         with torch.no_grad():
#             stats.accumulate(self.run)

#     def after_epoch(self):
#         print(self.train_stats)
#         print(self.valid_stats)


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
            stats.accumulate(self.learn)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


# # Version with runner
# class BatchTransformXCallback(Callback):
#     _order = 2
#     def __init__(self, tfm):
#         self.tfm = tfm

#     def begin_batch(self):
#         self.run.xb = self.tfm(self.xb)


class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.learn.xb = self.tfm(self.xb)


# # Version with runner
# class CudaCallback(Callback):
#     def begin_fit(self):
#         self.model.cuda()

#     def begin_batch(self):
#         self.run.xb = self.run.cuda()
#         self.run.yb = self.run.cuda()


class CudaCallback(Callback):
    def begin_fit(self):
        self.model.cuda()

    def begin_batch(self):
        self.learn.xb = self.learn.cuda()
        self.learn.yb = self.learn.cuda()


# # Version with runner
# class DeviceCallback(Callback):
#     def __init__(self):
#         self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#         print(f"self.device: {self.device}")

#     def begin_fit(self):
#         self.model.to(self.device)

#     def begin_batch(self):
#         self.run.xb = self.run.xb.to(self.device)
#         self.run.yb = self.run.yb.to(self.device)


class DeviceCallback(Callback):
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        print(f"self.device: {self.device}")

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        self.learn.xb = self.learn.xb.to(self.device)
        self.learn.yb = self.learn.yb.to(self.device)


class LearningrateFinder(Callback):
    _order = 1

    def __init__(self, max_iter=1000, min_lr=1e-2, max_lr=10):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss * 10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname = pname
        self.sched_funcs = sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs / self.epochs)

    def begin_batch(self):
        if self.in_train:
            self.set_param()


class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0):
        plt.plot(self.losses[: len(self.losses) - skip_last])

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(lrs[:n], losses[:n])


# # Version with runner
# class TrainEvalCallback(Callback):
#     def begin_fit(self):
#         self.run.n_epochs = 0.
#         self.run.n_iter = 0

#     def after_batch(self):
#         if not self.in_train:
#             return
#         self.run.n_epochs += 1./self.iters
#         self.run.n_iter += 1

#     def begin_epoch(self):
#         self.run.n_epochs = self.epoch
#         self.model.train()
#         self.run.in_train = True

#     def begin_validate(self):
#         self.model.eval()
#         self.run.in_train = False


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.learn.n_epochs = 0.0
        self.learn.n_iter = 0

    def after_batch(self):
        if not self.in_train:
            return
        self.learn.n_epochs += 1.0 / self.iters
        self.learn.n_iter += 1

    def begin_epoch(self):
        self.learn.n_epochs = self.epoch
        self.model.train()
        self.learn.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.learn.in_train = False


class WriteSubmissionCallback(Callback):
    pass


def annealer(func):
    def wrapper(start, end):
        return partial(func, start, end)

    return wrapper


@annealer
def sched_lin(start, end, pos):
    return start + (end - start) * pos


@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos):
    return start


@annealer
def sched_exp(start: int, end: int, pos: float) -> float:
    return start * (end / start) ** pos


def combine_scheds(pcts: list, scheds: list):
    """
    Combines different schedulers at different percentages of the training.

    Args:
        pcts: percentages at which a scheduler will trigger
        scheds: schedulers to be used
    """
    assert sum(pcts) == 1.0
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner
