"""Oscarmax attention

Clusters attention weights into groups with equal weight, regardless of index.

A Regularized Framework for Sparse and Structured Neural Attention
Vlad Niculae, Mathieu Blondel
https://arxiv.org/abs/1705.07704
"""

import numpy as np
import torch
from torch import nn
from torch import autograd as ta
from .isotonic import isotonic_regression
from .base import _BaseBatchProjection
from .sparsemax import SparsemaxFunction

def oscar_project_jv(y_hat, dout):
    y_hat = y_hat.detach().numpy()
    din = dout.clone().zero_()
    dout = dout.numpy()
    din_np = din.numpy()
    sign = np.sign(y_hat)
    y_hat = np.abs(y_hat)
    uniq, inv, counts = np.unique(y_hat, return_inverse=True, return_counts=True)
    n_unique = len(uniq)
    tmp = np.zeros((n_unique,), dtype=y_hat.dtype)
    np.add.at(tmp, inv, dout * sign)
    tmp /= counts
    tmp.take(inv, mode='clip', out=din_np)
    din_np *= sign
    return din

def prox_owl(v, w):
    v_abs = np.abs(v)
    ix = np.argsort(v_abs)[::-1]
    v_abs = v_abs[ix]
    v_abs = isotonic_regression(v_abs - w, y_min=0, increasing=False)
    inv_ix = np.zeros_like(ix)
    inv_ix[ix] = np.arange(len(v))
    v_abs = v_abs[inv_ix]
    return np.sign(v) * v_abs

def _oscar_weights(alpha, beta, size):
    w = np.arange(size - 1, -1, -1, dtype=np.float32)
    w *= beta
    w += alpha
    return w

def oscar_project(self, x, alpha=0, beta=1):
        x_np = x.detach().numpy().copy()
        weights = _oscar_weights(self.alpha, self.beta, x_np.shape[0])
        y_hat_np = prox_owl(x_np, weights)
        y_hat = torch.from_numpy(y_hat_np)
        return y_hat


class OscarProxFunction(_BaseBatchProjection):


class Oscarmax(nn.Module):
    
    def forward(self, x, beta=1, lengths=None):
        oscar_prox = OscarProxFunction(beta=beta)
        sparsemax = SparsemaxFunction()
        return sparsemax.apply(oscar_prox.apply(x, lengths), lengths)

