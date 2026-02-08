"""
Hyperbolic Operations for Poincaré Ball Model

This module implements numerically stable operations in hyperbolic space
using the Poincaré ball model. All operations include extensive safety checks
to prevent numerical instabilities (NaNs, boundary violations).

Key Operations:
- exp_map: Exponential map (tangent space → manifold)
- log_map: Logarithmic map (manifold → tangent space)
- mobius_add: Möbius addition (gyrovector addition)
- poincare_distance: Hyperbolic distance between points
- project_onto_ball: Safe projection ensuring points stay in ball
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def project_onto_ball(
    x: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Project points onto Poincaré ball with safety margin.

    Args:
        x: Points to project [*, dim]
        c: Curvature (positive for hyperbolic space)
        eps: Safety margin from boundary

    Returns:
        Projected points safely within ball [*, dim]
    """
    c = max(c, 1e-6)  # Prevent division by zero
    max_norm = (1.0 / math.sqrt(c)) - eps

    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
    cond = norm > max_norm

    projected = torch.where(
        cond,
        x * (max_norm / norm),
        x
    )

    return projected


def mobius_add(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Möbius addition in Poincaré ball.

    Formula: x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)

    Args:
        x: First point [*, dim]
        y: Second point [*, dim]
        c: Curvature

    Returns:
        Result of Möbius addition [*, dim]
    """
    c = max(c, 1e-6)

    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    numerator = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denominator = 1 + 2 * c * xy + (c ** 2) * x_sq * y_sq

    # Prevent division by zero
    denominator = denominator.clamp(min=1e-8)

    result = numerator / denominator

    # Always project to ensure numerical stability
    return project_onto_ball(result, c, eps=1e-5)


def exp_map(
    v: torch.Tensor,
    x: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Exponential map: tangent space at x → Poincaré ball.

    Maps tangent vectors to the manifold using the exponential map.

    Formula: exp_x(v) = x ⊕_c (tanh(√c λ_x ||v|| / 2) * v / (√c ||v||))
    where λ_x = 2 / (1 - c||x||²) is the conformal factor

    Args:
        v: Tangent vector at x [*, dim]
        x: Base point on manifold [*, dim]
        c: Curvature

    Returns:
        Point on manifold [*, dim]
    """
    c = max(c, 1e-6)
    sqrt_c = math.sqrt(c)

    # Conformal factor λ_x
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    x_norm_sq = x_norm_sq.clamp(max=1.0 / c - 1e-4)  # Safety: keep away from boundary
    lambda_x = 2.0 / (1.0 - c * x_norm_sq + 1e-6)

    # Norm of tangent vector
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)

    # Scaled tangent vector
    coeff = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm)
    scaled_v = coeff * v

    # Möbius addition
    result = mobius_add(x, scaled_v, c)

    return result


def log_map(
    y: torch.Tensor,
    x: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Logarithmic map: Poincaré ball → tangent space at x.

    Maps points on the manifold back to tangent vectors.

    Formula: log_x(y) = (2 / (√c λ_x)) * artanh(√c ||-x ⊕_c y||) * (-x ⊕_c y) / ||-x ⊕_c y||

    Args:
        y: Point on manifold [*, dim]
        x: Base point on manifold [*, dim]
        c: Curvature

    Returns:
        Tangent vector at x [*, dim]
    """
    c = max(c, 1e-6)
    sqrt_c = math.sqrt(c)

    # Compute -x ⊕_c y
    neg_x = -x
    diff = mobius_add(neg_x, y, c)

    # Norm of difference
    diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-10, max=1.0 / sqrt_c - 1e-5)

    # Conformal factor λ_x
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    x_norm_sq = x_norm_sq.clamp(max=1.0 / c - 1e-4)
    lambda_x = 2.0 / (1.0 - c * x_norm_sq + 1e-6)

    # Log map formula
    atanh_arg = sqrt_c * diff_norm
    atanh_arg = atanh_arg.clamp(min=-1.0 + 1e-5, max=1.0 - 1e-5)  # Safety for atanh

    coeff = (2.0 / (sqrt_c * lambda_x)) * torch.atanh(atanh_arg) / diff_norm

    result = coeff * diff

    return result


def exp_map_zero(
    v: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Exponential map at origin (simplified version).

    When the base point is origin, the formula simplifies significantly.

    Formula: exp_0(v) = tanh(√c ||v||) / (√c ||v||) * v

    Args:
        v: Tangent vector at origin [*, dim]
        c: Curvature

    Returns:
        Point on manifold [*, dim]
    """
    c = max(c, 1e-6)
    sqrt_c = math.sqrt(c)

    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)

    coeff = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    result = coeff * v

    return project_onto_ball(result, c, eps=1e-5)


def log_map_zero(
    x: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Logarithmic map at origin (simplified version).

    Formula: log_0(x) = artanh(√c ||x||) / (√c ||x||) * x

    Args:
        x: Point on manifold [*, dim]
        c: Curvature

    Returns:
        Tangent vector at origin [*, dim]
    """
    c = max(c, 1e-6)
    sqrt_c = math.sqrt(c)

    x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10, max=1.0 / sqrt_c - 1e-5)

    atanh_arg = sqrt_c * x_norm
    atanh_arg = atanh_arg.clamp(min=-1.0 + 1e-5, max=1.0 - 1e-5)

    coeff = torch.atanh(atanh_arg) / (sqrt_c * x_norm)
    result = coeff * x

    return result


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Hyperbolic distance in Poincaré ball.

    Formula: d_c(x, y) = (2 / √c) * artanh(√c ||-x ⊕_c y||)

    Args:
        x: First point [*, dim]
        y: Second point [*, dim]
        c: Curvature

    Returns:
        Hyperbolic distance [*]
    """
    c = max(c, 1e-6)
    sqrt_c = math.sqrt(c)

    # Compute -x ⊕_c y
    neg_x = -x
    diff = mobius_add(neg_x, y, c)

    # Norm of difference
    diff_norm = torch.norm(diff, dim=-1).clamp(min=1e-10, max=1.0 / sqrt_c - 1e-5)

    # Distance formula
    atanh_arg = sqrt_c * diff_norm
    atanh_arg = atanh_arg.clamp(min=-1.0 + 1e-5, max=1.0 - 1e-5)

    distance = (2.0 / sqrt_c) * torch.atanh(atanh_arg)

    return distance


def pairwise_poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Pairwise hyperbolic distances between two sets of points.

    Args:
        x: First set of points [*, n, dim]
        y: Second set of points [*, m, dim]
        c: Curvature

    Returns:
        Pairwise distances [*, n, m]
    """
    # Expand dimensions for broadcasting
    x_exp = x.unsqueeze(-2)  # [*, n, 1, dim]
    y_exp = y.unsqueeze(-3)  # [*, 1, m, dim]

    # Compute pairwise distances
    distances = poincare_distance(x_exp, y_exp, c)

    return distances


def parallel_transport(
    v: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """
    Parallel transport of tangent vector v from x to y.

    Args:
        v: Tangent vector at x [*, dim]
        x: Source point [*, dim]
        y: Target point [*, dim]
        c: Curvature

    Returns:
        Tangent vector at y [*, dim]
    """
    c = max(c, 1e-6)

    # Compute gyration factor
    neg_x = -x
    diff = mobius_add(neg_x, y, c)

    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    lambda_x = 2.0 / (1.0 - c * x_sq + 1e-6)
    lambda_y = 2.0 / (1.0 - c * y_sq + 1e-6)

    # Parallel transport formula
    transported = (lambda_x / lambda_y) * v

    return transported


# Utility functions

def random_ball_points(
    *shape,
    c: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Generate random points uniformly in Poincaré ball.

    Args:
        shape: Shape of output tensor
        c: Curvature
        device: Device to place tensor on
        dtype: Data type

    Returns:
        Random points in ball [*shape]
    """
    # Generate random points in Euclidean space
    points = torch.randn(*shape, device=device, dtype=dtype)

    # Project onto ball with some margin
    return project_onto_ball(points * 0.1, c, eps=1e-3)


def check_bounds(
    x: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-5
) -> dict:
    """
    Check if points are within valid bounds.

    Args:
        x: Points to check [*, dim]
        c: Curvature
        eps: Tolerance

    Returns:
        Dictionary with diagnostics
    """
    c = max(c, 1e-6)
    max_norm = 1.0 / math.sqrt(c) - eps

    norms = torch.norm(x, dim=-1)

    return {
        'max_norm': norms.max().item(),
        'min_norm': norms.min().item(),
        'mean_norm': norms.mean().item(),
        'violations': (norms > max_norm).sum().item(),
        'nan_count': torch.isnan(x).sum().item(),
        'inf_count': torch.isinf(x).sum().item(),
    }
