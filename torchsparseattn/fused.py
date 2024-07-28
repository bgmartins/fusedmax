from __future__ import division
import torch
from torch import nn
from torch import autograd as ta
import warnings
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

def fused_project(x, alpha=1):
    x_np = x.detach().numpy().copy()
    prox_tv1d(x_np, alpha)
    y_hat = torch.from_numpy(x_np)
    return y_hat

def fused_project_jv(dout, y_hat):
    dout = dout.clone()
    _inplace_fused_prox_jv(y_hat.detach().numpy(), dout.numpy())
    return dout

class FusedProxFunction(ta.Function):
        
    @staticmethod
    def forward(ctx, x, lengths=None):
        requires_squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            requires_squeeze = True
        n_samples, max_dim = x.size()
        has_lengths = True
        if lengths is None:
            has_lengths = False
            lengths = [max_dim] * n_samples
        y_star = x.new()
        y_star.resize_as_(x)
        y_star.zero_()
        for i in range(n_samples): y_star[i, :lengths[i]] = fused_project(x[i, :lengths[i]])
        if requires_squeeze: y_star = y_star.squeeze()
        ctx.mark_non_differentiable(y_star)
        if has_lengths:
            ctx.mark_non_differentiable(lengths)
            ctx.save_for_backward(y_star, lengths)
        else: ctx.save_for_backward(y_star)
        return y_star
    
    @staticmethod
    def backward(ctx, dout):
        if not ctx.needs_input_grad[0]: return None
        if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]: raise ValueError("Cannot differentiate {} w.r.t. the sequence lengths".format(ctx.__name__))
        saved = ctx.saved_tensors
        if len(saved) == 2: y_star, lengths = saved
        else:
            y_star, = saved
            lengths = None
        requires_squeeze = False
        if y_star.dim() == 1:
            y_star = y_star.unsqueeze(0)
            dout = dout.unsqueeze(0)
            requires_squeeze = True
        n_samples, max_dim = y_star.size()
        din = dout.new()
        din.resize_as_(y_star)
        din.zero_()
        if lengths is None: lengths = [max_dim] * n_samples
        for i in range(n_samples): din[i, :lengths[i]] = fused_project_jv(dout[i, :lengths[i]], y_star[i, :lengths[i]])
        if requires_squeeze: din = din.squeeze()
        return din, None

class Fusedmax(nn.Module):
    
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(Fusedmax, self).__init__()

    def forward(self, x, lengths=None):
        fused_prox = FusedProxFunction(self.alpha)
        sparsemax = SparsemaxFunction()
        return sparsemax.apply(fused_prox.apply(x, lengths), lengths)
