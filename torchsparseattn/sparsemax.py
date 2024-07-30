from __future__ import division
import numpy as np
import torch
from torch import nn
from torch import autograd as ta

def project(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).to(dtype=v.dtype).to(v.device)
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)
    return w

def project_jv(dout, w_star):
    supp = w_star > 0
    masked = dout.masked_select(supp)
    nnz = supp.to(dtype=dout.dtype).sum()
    masked -= masked.sum() / nnz
    out = dout.new(dout.size()).zero_()
    out[supp] = masked
    return(out)   
     
class SparsemaxFunction(ta.Function):
        
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
        for i in range(n_samples): y_star[i, :lengths[i]] = project(x[i, :lengths[i]])
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
        for i in range(n_samples): din[i, :lengths[i]] = project_jv(dout[i, :lengths[i]], y_star[i, :lengths[i]])
        if requires_squeeze: din = din.squeeze()
        return din, None

class Sparsemax(nn.Module):

    def forward(self, x, lengths=None):
        sparsemax = SparsemaxFunction()
        return sparsemax.apply(x, lengths)
