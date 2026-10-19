import torch
from torch import Tensor


class _GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output * ctx.scale, None


def scale_gradient(tensor: Tensor, scale: float) -> Tensor:
    """ Identity in the forward pass, multiplies the gradient by *scale* in the backward pass.

    A scale of 1.0 returns the tensor unchanged and a scale of 0.0 detaches it, so neither
    case adds a custom autograd node to the graph.
    """
    if scale == 1.0:
        return tensor

    if scale == 0.0:
        return tensor.detach()

    return _GradientScale.apply(tensor, scale)
