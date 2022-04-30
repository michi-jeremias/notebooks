import torch
from functools import partial
from dltools.functions import listify
from matplotlib import pyplot as plt


class ListContainer:
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.items[idx]
        if isinstance(idx[0], bool):
            assert len(idx) == len(self)
            return [o for m, o in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del self.items[i]

    def __repr__(self):
        res = f"{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            res = res[:-1] + "...]"
        return res


class Hook:
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(ListContainer):
    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = ([], [])
    means, stds = hook.stats
    means.append(outp.data.mean())
    stds.append(outp.data.std())


def append_stats_hist(hook, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    means, stds, hists = hook.stats
    means.append(outp.data.mean().cpu())
    stds.append(outp.data.std().cpu())
    # hists.append(outp.data.cpu().histc(40, 0, 10))  # histc isn't implemented on the GPU
    hists.append(
        outp.data.cpu().histc(40, -10, 10)
    )  # histc isn't implemented on the GPU


def children(m):
    return list(m.children())


def get_hist(h):
    return torch.stack(h.stats[2]).t().float().log1p()


def get_min(h):
    # Assume first two bins of the histogram contain values close to 0 (if the histogram's
    # lower boundary is 0. If the histogram is symmetric around 0 it's the middle bins).
    # Show relative amount of data in these bins compared to all bins.
    # This gives a ratio of values close to 0.
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[19:22].sum(0) / h1.sum(0)


def telemetrize(hooks, xlim):
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    for ax, h in zip(axes.flatten(), hooks[:4]):
        ax.set_ylabel("Hist bin")
        ax.set_xlabel("Iteration")
        ax.plot([0, xlim], [20.0, 20.0], c="r", linewidth=1)
        ax.imshow(get_hist(h), aspect="auto", origin="lower")
    plt.tight_layout()

    fig, axes = plt.subplots(2, 2, figsize=(17, 6))
    for ax, h in zip(axes.flatten(), hooks[:4]):
        ax.set_ylabel("%/100 zeros in activations")
        ax.set_xlabel("Iteration")
        ax.plot([0, xlim], [1.0, 1.0], c="r", linewidth=1)
        ax.plot(get_min(h))
        ax.set_ylim(0, 1.1)
    plt.tight_layout()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 4))
    for h in hooks[:-1]:
        ms, ss, hists = h.stats
        ax0.plot(ms[:10])
        ax1.plot(ss[:10])
        h.remove()
    plt.legend(range(6))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 4))
    for h in hooks[:-1]:
        ms, ss, hists = h.stats
        ax0.plot(ms)
        ax1.plot(ss)
