import gzip
import pickle
import re
from functools import partial
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from fastai import datasets
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch import nn, optim, tensor
from torch.utils.data import DataLoader

# from dltools.learner import Learner
from dltools.data import TypeSelector
from dltools.layer import Lambda, flatten

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"


def activation_stats(model, data, sh=None) -> None:
    """
    Prints mean and std of activations after each layer.
    sh is a list that defines a shape for view. Default for sh is x.shape (view unchanged).
    """
    ms, ss = [], []
    x, y = next(iter(data.train_dl))
    if not sh:
        sh = x.shape
    x = x.view(sh)
    for i, l in enumerate(model.cpu()):
        ms.append(model.cpu()[:i](x).mean())
        ss.append(model.cpu()[:i](x).std())
    print(f"Batchsize:\t{data.train_dl.batch_size}")
    print(f"Means:\t\t{torch.tensor(ms)}")
    print(f"Stds:\t\t{torch.tensor(ss)}\n")


def camel2snake(name):
    """
    Converts a CamelCaseString into a snake_case_string.
    """
    _camel_re1 = re.compile("(.)([A-Z][a-z]+)")
    _camel_re2 = re.compile("([a-z0-9])([A-Z])")
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


# def create_learner(model_func, loss_func, data):
#     """
#     Returns a Learner object. A Learner is a container class for a model, a
#     loss function and a data object.
#     """
#     return Learner(*model_func(data), loss_func, data)


def get_data():
    path = datasets.download_data(MNIST_URL, ext=".gz")
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return map(tensor, (x_train, y_train, x_valid, y_valid))


def get_dls(train_ds, valid_ds, batch_size, **kwargs):
    """
    Returns a dataloader for each dataset. Batchsize for the DL of the
    validation set is double the size because more memory is available due to
    not having gradients stored.
    """
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False, **kwargs),
    )


def get_model(data: tensor, lr=0.5, nh=50):
    """
    Returns a linear model and a SGD optimizer.
    """
    # m = data.train_ds.x.shape[1]
    m = data.train_ds.tensors[0].shape[1]
    model = nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, data.c))
    return model, optim.SGD(model.parameters(), lr)


def get_model_func(lr=0.5):
    return partial(get_model, lr=lr)


def init_cnn(model, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    for layer in model:
        if isinstance(layer, nn.Sequential):
            f(layer[0].weight, a=0.1)
            layer[0].bias.data.zero_()


def init_cnn_(model, f):
    if isinstance(model, nn.Conv2d):
        f(model.weight, a=0.1)
        if getattr(model, "bias", None) is not None:
            model.bias.data.zero_()
    for layer in model.children():
        init_cnn_(layer, f)


def init_lin_(model, f):
    if isinstance(model, nn.Linear):
        f(model.weight, a=0.1)
        if getattr(model, "bias", None) is not None:
            model.bias.data.zero_()
    for layer in model.children():
        init_lin_(layer, f)


def init_lin(m, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_lin_(m, f)


def listify(o):
    """
    Converts everything into a list.
    """
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


def normalize(x, m, s):
    """
    Subtracts the mean m of x and divides by the standard deviation s.
    I.e. normalizing x.
    """
    return (x - m) / s


def normalize_to(train: tensor, valid: tensor, test=None):
    """
    Normalize training and validation data, using mean and std of the training
    data.
    """
    m, s = train.mean(), train.std()
    if test is None:
        return normalize(train, m, s), normalize(valid, m, s)
    else:
        return normalize(train, m, s), normalize(valid, m, s), normalize(test, m, s)


def proc_df(cont_names, cat_names, *dataframes):
    df_all = pd.concat(dataframes, axis=0)
    df_all[cont_names] = df_all[cont_names].astype("float64")
    df_all[cat_names] = df_all[cat_names].astype("category")

    for df in dataframes:
        df[cont_names] = df[cont_names].astype("float64")
        df[cat_names] = df[cat_names].astype("category")

    ordered_cont_names = [col for col in df_all.columns if col in cont_names]
    # nested list comprehension to get column names for onehot encoded categoy columns
    ordered_onehot_names = [
        f"{i}_{cat}" for cat in df_all[cat_names] for i in range(df_all[cat].nunique())
    ]
    names = ordered_cont_names + ordered_onehot_names

    # preprocessing pipeline
    pp = make_pipeline(
        FeatureUnion(
            transformer_list=[
                (
                    "numeric_features",
                    make_pipeline(
                        TypeSelector("float64"),
                        SimpleImputer(strategy="median"),
                        # QuantileTransformer()
                        # StandardScaler()
                    ),
                ),
                (
                    "categorical_features",
                    make_pipeline(
                        TypeSelector("category"),
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(categories="auto", handle_unknown="ignore"),
                    ),
                ),
            ]
        )
    )
    pp.fit(df_all)

    return (
        pd.DataFrame(pp.transform(df).toarray(), columns=names) for df in dataframes
    )


def rmsle(x: tensor, y: tensor) -> tensor:
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return torch.sqrt(((torch.log(x + 1) - torch.log(y + 1)) ** 2).mean())


def test(val_a, val_b, cmp, cname=None):
    """
    Compares two values val_a and val_b according to an operator cmb
    """
    if cname is None:
        cname = cmp.__name__
    assert cmp(val_a, val_b), f"{cname}:\n{val_a}\n{val_b}"


def view_tfm(*size):
    def _inner(x):
        return x.view(*((-1,) + size))

    return _inner
