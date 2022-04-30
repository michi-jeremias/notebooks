import torch
from dltools.functions import init_cnn

torch.Tensor.ndim = property(lambda x: len(x.shape))
