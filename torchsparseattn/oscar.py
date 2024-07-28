import numpy as np
import torch
from torch import nn
from torch import autograd as ta
from .isotonic import isotonic_regression
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

class OscarProxFunction(ta.Function):
        
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
        for i in range(n_samples): y_star[i, :lengths[i]] = oscar_project(x[i, :lengths[i]])
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
        for i in range(n_samples): din[i, :lengths[i]] = oscar_project_jv(dout[i, :lengths[i]], y_star[i, :lengths[i]])
        if requires_squeeze: din = din.squeeze()
        return din, None

class Oscarmax(nn.Module):
    def __init__(self, beta=1):
        self.beta = beta
        super(Oscarmax, self).__init__()

    def forward(self, x, lengths=None):
        oscar_prox = OscarProxFunction(beta=self.beta)
        sparsemax = SparsemaxFunction()
        return sparsemax.apply(oscar_prox.apply(x, lengths), lengths)
