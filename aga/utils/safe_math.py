"""
Safe Mathematical Operations with Bounded Gradients

Provides numerically stable versions of mathematical functions
that are safe for backpropagation in hyperbolic geometry.
"""

import torch
import torch.nn as nn


class SafeAtanh(torch.autograd.Function):
    """
    Safe arctanh with bounded gradients.

    The standard atanh has gradient 1/(1-x^2) which explodes as x→1.
    This implementation clamps THE GRADIENT ITSELF to prevent NaN propagation.
    """

    @staticmethod
    def forward(ctx, x, eps=1e-5):
        # Forward pass: Clamp input for valid math
        x = x.clamp(-1 + eps, 1 - eps)
        ctx.save_for_backward(x)
        return torch.atanh(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Standard derivative: 1 / (1 - x^2)
        # PROBLEM: As x -> 1, this -> Infinity

        grad_input = 1 / (1 - x ** 2)

        # --- THE FIX: CLAMP THE GRADIENT ITSELF ---
        # Limit the maximum slope of the function
        grad_input = grad_input.clamp(max=100.0)

        return grad_input * grad_output, None


class SafeTanh(torch.autograd.Function):
    """
    Safe tanh with bounded gradients.

    Standard tanh is usually fine, but we add safeguards
    for consistency with atanh.
    """

    @staticmethod
    def forward(ctx, x):
        result = torch.tanh(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        tanh_x, = ctx.saved_tensors
        # Standard gradient: 1 - tanh^2(x)
        # Clamp to prevent underflow
        grad_input = grad_output * (1 - tanh_x ** 2).clamp(min=1e-10)
        return grad_input


def safe_atanh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable atanh with bounded gradients.

    Args:
        x: Input tensor

    Returns:
        atanh(x) with safe backpropagation
    """
    return SafeAtanh.apply(x)


def safe_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable tanh with bounded gradients.

    Args:
        x: Input tensor

    Returns:
        tanh(x) with safe backpropagation
    """
    return SafeTanh.apply(x)


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = True, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute norm with minimum threshold to prevent division by zero.

    Args:
        x: Input tensor
        dim: Dimension along which to compute norm
        keepdim: Whether to keep dimension
        eps: Minimum value for norm

    Returns:
        Clamped norm
    """
    return torch.norm(x, dim=dim, keepdim=keepdim).clamp(min=eps)
