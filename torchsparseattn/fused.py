from __future__ import division
import torch
from torch import nn
from torch import autograd as ta
import warnings
from .base import _BaseBatchProjection
from .sparsemax import SparsemaxFunction
from ._fused import prox_tv1d

def _inplace_fused_prox_jv_slow(y_hat, dout):
    n_features = len(dout)
    for i in range(n_features + 1):
        if i in (0, n_features) or y_hat[i] != y_hat[i - 1]:
            if i > 0: dout[last_ix:i] = acc / n
            if i < n_features:
                last_ix = i
                acc = dout[i]
                n = 1
        else:
            acc += dout[i]
            n += 1
    return dout

try:
    from ._fused_jv import _inplace_fused_prox_jv
except ImportError:
    warnings.warn("Could not import cython implementation of fused backward pass. Slow implementation used instead.")
    _inplace_fused_prox_jv = _inplace_fused_prox_jv_slow

def fused_prox_jv_slow(y_hat, dout):
    dout = dout.clone()
    _inplace_fused_prox_jv_slow(y_hat, dout)
    return dout

def fused_prox_jv_fast(y_hat, dout):
    dout = dout.clone()
    _inplace_fused_prox_jv(y_hat.detach().numpy(), dout.numpy())
    return dout

class FusedProxFunction(_BaseBatchProjection):

    @staticmethod
    def project(x, alpha=1):
        x_np = x.detach().numpy().copy()
        prox_tv1d(x_np, alpha)
        y_hat = torch.from_numpy(x_np)
        return y_hat

    @staticmethod
    def project_jv(dout, y_hat):
        dout = dout.clone()
        _inplace_fused_prox_jv(y_hat.detach().numpy(), dout.numpy())
        return dout

class Fusedmax(nn.Module):

    @staticmethod
    def forward(x, alpha=1, lengths=None):
        fused_prox = FusedProxFunction(alpha)
        sparsemax = SparsemaxFunction()
        return sparsemax.apply(fused_prox.apply(x, lengths), lengths)
