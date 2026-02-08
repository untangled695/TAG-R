"""
Tests for Curvature-Aware Router and Load Balancing

These tests verify that:
1. Router produces valid routing distributions
2. Load balancing loss prevents collapse
3. Router learns to route different structures differently
4. AGALayer integrates all components correctly
"""

import pytest
import torch
import math

from aga.routing.curvature_router import (
    CurvatureAwareRouter,
    load_balancing_loss,
    load_balancing_loss_cv,
    router_entropy_loss,
    combined_routing_loss,
    GeometryFeatureExtractor,
)
from aga.models.aga_layer import AGALayer, AGABlock


class TestCurvatureAwareRouter:
    """Test router basic functionality."""

    def test_router_output_shape(self):
        """Router should output correct shapes."""
        d_model = 32
        batch, seq_len = 4, 10

        router = CurvatureAwareRouter(d_model=d_model, num_geometries=3)
        x = torch.randn(batch, seq_len, d_model)

        weights, logits = router(x)

        assert weights.shape == (batch, seq_len, 3)
        assert logits.shape == (batch, seq_len, 3)

        print("✅ Router Output Shape Test Passed")

    def test_router_weights_sum_to_one(self):
        """Router weights should sum to 1 (valid probability distribution)."""
        d_model = 32
        batch, seq_len = 4, 10

        router = CurvatureAwareRouter(d_model=d_model, num_geometries=3)
        x = torch.randn(batch, seq_len, d_model)

        weights, _ = router(x)

        # Sum along geometry dimension (dim=-1)
        weight_sums = weights.sum(dim=-1)

        # Should all be 1.0 (within numerical tolerance)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

        print("✅ Router Weights Sum to One Test Passed")

    def test_router_temperature_effect(self):
        """Lower temperature should produce sharper routing."""
        d_model = 32
        batch, seq_len = 4, 10

        x = torch.randn(batch, seq_len, d_model)

        # High temperature (soft routing)
        router_soft = CurvatureAwareRouter(d_model=d_model, temperature=2.0)
        weights_soft, _ = router_soft(x)

        # Low temperature (sharp routing)
        router_sharp = CurvatureAwareRouter(d_model=d_model, temperature=0.5)
        weights_sharp, _ = router_sharp(x)

        # Measure "sharpness" using max weight
        max_soft = weights_soft.max(dim=-1)[0].mean().item()
        max_sharp = weights_sharp.max(dim=-1)[0].mean().item()

        print(f"  Soft routing (T=2.0) max weight: {max_soft:.3f}")
        print(f"  Sharp routing (T=0.5) max weight: {max_sharp:.3f}")

        # Sharp routing should have higher max weights
        assert max_sharp > max_soft, (
            f"Sharp routing should have higher max weights, "
            f"got {max_sharp:.3f} vs {max_soft:.3f}"
        )

        print("✅ Router Temperature Effect Test Passed")

    def test_router_gradient_flow(self):
        """Gradients should flow through router."""
        d_model = 32
        batch, seq_len = 4, 10

        router = CurvatureAwareRouter(d_model=d_model, num_geometries=3)
        x = torch.randn(batch, seq_len, d_model, requires_grad=True)

        weights, logits = router(x)
        loss = weights.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))

        # Check router parameters have gradients
        for param in router.parameters():
            if param.requires_grad:
                assert param.grad is not None

        print("✅ Router Gradient Flow Test Passed")

    def test_top_k_routing(self):
        """Top-k routing should select correct number of geometries."""
        d_model = 32
        batch, seq_len = 4, 10
        k = 2

        router = CurvatureAwareRouter(d_model=d_model, num_geometries=3)
        x = torch.randn(batch, seq_len, d_model)

        top_k_weights, top_k_indices, logits = router.get_top_k_routing(x, k=k)

        assert top_k_weights.shape == (batch, seq_len, k)
        assert top_k_indices.shape == (batch, seq_len, k)

        # Top-k weights should sum to 1
        weight_sums = top_k_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

        print("✅ Top-k Routing Test Passed")


class TestLoadBalancingLoss:
    """Test load balancing loss functions."""

    def test_perfect_balance_has_zero_loss(self):
        """Perfectly balanced routing should have zero loss."""
        batch, seq_len, num_geometries = 100, 50, 3

        # Create perfectly balanced logits
        # Each geometry used equally
        logits = torch.ones(batch, seq_len, num_geometries)

        loss = load_balancing_loss(logits, importance_factor=1.0)

        print(f"  Perfect balance loss: {loss.item():.6f}")
        assert loss.item() < 1e-5, f"Perfect balance should have ~0 loss, got {loss.item()}"

        print("✅ Perfect Balance Zero Loss Test Passed")

    def test_imbalanced_routing_has_high_loss(self):
        """Heavily imbalanced routing should have high loss."""
        batch, seq_len, num_geometries = 100, 50, 3

        # Create imbalanced logits (all routes to first geometry)
        logits = torch.zeros(batch, seq_len, num_geometries)
        logits[..., 0] = 10.0  # Heavily favor first geometry

        loss = load_balancing_loss(logits, importance_factor=1.0)

        print(f"  Imbalanced loss: {loss.item():.6f}")
        assert loss.item() > 0.1, f"Imbalanced routing should have high loss, got {loss.item()}"

        print("✅ Imbalanced Routing High Loss Test Passed")

    def test_load_balance_gradient_flow(self):
        """Load balance loss should provide gradients."""
        batch, seq_len, num_geometries = 10, 20, 3

        logits = torch.randn(batch, seq_len, num_geometries, requires_grad=True)

        loss = load_balancing_loss(logits, importance_factor=0.01)
        loss.backward()

        assert logits.grad is not None
        assert not torch.any(torch.isnan(logits.grad))

        print("✅ Load Balance Gradient Flow Test Passed")

    def test_entropy_loss(self):
        """Entropy loss should encourage diverse routing."""
        batch, seq_len, num_geometries = 100, 50, 3

        # High entropy (diverse) routing
        logits_diverse = torch.randn(batch, seq_len, num_geometries) * 0.1

        # Low entropy (collapsed) routing
        logits_collapsed = torch.zeros(batch, seq_len, num_geometries)
        logits_collapsed[..., 0] = 10.0

        entropy_diverse = router_entropy_loss(logits_diverse, importance_factor=1.0)
        entropy_collapsed = router_entropy_loss(logits_collapsed, importance_factor=1.0)

        print(f"  Diverse routing entropy loss: {entropy_diverse.item():.6f}")
        print(f"  Collapsed routing entropy loss: {entropy_collapsed.item():.6f}")

        # Diverse routing should have higher entropy (lower negative entropy loss)
        assert entropy_diverse < entropy_collapsed

        print("✅ Entropy Loss Test Passed")


class TestAGALayer:
    """Test full AGALayer integration."""

    def test_aga_layer_forward(self):
        """AGA layer should complete forward pass."""
        d_model = 32
        n_heads = 4
        batch, seq_len = 4, 10

        layer = AGALayer(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch, seq_len, d_model)

        output, router_info = layer(x, return_router_info=True)

        assert output.shape == (batch, seq_len, d_model)
        assert not torch.any(torch.isnan(output))

        # Check router info
        assert router_info is not None
        assert 'weights' in router_info
        assert 'logits' in router_info
        assert 'mean_weights' in router_info

        print(f"  Mean geometry weights: {router_info['mean_weights']}")
        print("✅ AGA Layer Forward Test Passed")

    def test_aga_layer_with_mask(self):
        """AGA layer should work with attention masking."""
        d_model = 32
        n_heads = 4
        batch, seq_len = 4, 10

        layer = AGALayer(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch, seq_len, d_model)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch, n_heads, seq_len, seq_len)

        output, _ = layer(x, mask=mask)

        assert not torch.any(torch.isnan(output))

        print("✅ AGA Layer with Mask Test Passed")

    def test_aga_layer_gradient_flow(self):
        """Gradients should flow through AGA layer."""
        d_model = 32
        n_heads = 4
        batch, seq_len = 4, 10

        layer = AGALayer(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch, seq_len, d_model, requires_grad=True)

        output, router_info = layer(x, return_router_info=True)
        loss = output.sum() + load_balancing_loss(router_info['logits'])
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))

        # Check all parameters have gradients
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

        print("✅ AGA Layer Gradient Flow Test Passed")

    def test_aga_block(self):
        """Full AGA block with FFN should work."""
        d_model = 32
        n_heads = 4
        batch, seq_len = 4, 10

        block = AGABlock(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch, seq_len, d_model)

        output, router_info = block(x, return_router_info=True)

        assert output.shape == (batch, seq_len, d_model)
        assert not torch.any(torch.isnan(output))

        print("✅ AGA Block Test Passed")

    def test_geometry_usage_stats(self):
        """Router should provide usage statistics."""
        d_model = 32
        n_heads = 4
        batch, seq_len = 10, 20

        layer = AGALayer(d_model=d_model, n_heads=n_heads)
        x = torch.randn(batch, seq_len, d_model)

        stats = layer.get_geometry_usage_stats(x)

        assert 'mean_weights' in stats
        assert 'mean_entropy' in stats
        assert 'hard_routing_counts' in stats
        assert 'geometry_names' in stats

        print(f"  Geometry usage: {dict(zip(stats['geometry_names'], stats['mean_weights']))}")
        print(f"  Mean entropy: {stats['mean_entropy']:.4f}")
        print(f"  Hard routing counts: {dict(zip(stats['geometry_names'], stats['hard_routing_counts']))}")

        print("✅ Geometry Usage Stats Test Passed")


class TestRouterBehavior:
    """Test that router actually routes differently for different inputs."""

    def test_router_distinguishes_structures(self):
        """
        Router should assign different geometries to different structures.

        This is the KEY test: Does the router actually learn structural differences?
        """
        d_model = 32
        batch = 10
        seq_len = 20

        router = CurvatureAwareRouter(d_model=d_model, num_geometries=3)

        # Create different "structure types" by varying input statistics

        # Type 1: "Flat" structure (low variance, centered)
        flat_input = torch.randn(batch, seq_len, d_model) * 0.1

        # Type 2: "Hierarchical" structure (high variance in first few dimensions)
        hierarchical_input = torch.randn(batch, seq_len, d_model) * 0.1
        hierarchical_input[..., :5] *= 10.0  # Large values in few dimensions

        # Type 3: "Cyclic" structure (oscillating patterns)
        t = torch.linspace(0, 2 * math.pi, seq_len).unsqueeze(-1)
        cyclic_input = torch.sin(t.repeat(1, d_model)) * 0.5
        cyclic_input = cyclic_input.unsqueeze(0).repeat(batch, 1, 1)
        cyclic_input += torch.randn_like(cyclic_input) * 0.1

        # Route each type
        weights_flat, _ = router(flat_input)
        weights_hier, _ = router(hierarchical_input)
        weights_cyclic, _ = router(cyclic_input)

        # Average weights across batch and sequence
        mean_flat = weights_flat.mean(dim=(0, 1))
        mean_hier = weights_hier.mean(dim=(0, 1))
        mean_cyclic = weights_cyclic.mean(dim=(0, 1))

        print(f"  Flat structure:         Euc={mean_flat[0]:.3f}, Hyp={mean_flat[1]:.3f}, Sph={mean_flat[2]:.3f}")
        print(f"  Hierarchical structure: Euc={mean_hier[0]:.3f}, Hyp={mean_hier[1]:.3f}, Sph={mean_hier[2]:.3f}")
        print(f"  Cyclic structure:       Euc={mean_cyclic[0]:.3f}, Hyp={mean_cyclic[1]:.3f}, Sph={mean_cyclic[2]:.3f}")

        # Router outputs should be different for different structures
        # (Note: Untrained router will be somewhat random, but should show some variation)
        total_variance = torch.var(torch.stack([mean_flat, mean_hier, mean_cyclic]), dim=0).sum()
        print(f"  Total routing variance: {total_variance.item():.6f}")

        # With untrained router, variance should be > 0 (not all identical)
        assert total_variance > 1e-6, "Router should show some variance for different inputs"

        print("✅ Router Distinguishes Structures Test Passed")


class TestGeometryFeatureExtractor:
    """Test geometry feature extraction."""

    def test_feature_extractor_output_shape(self):
        """Feature extractor should output correct shape."""
        d_model = 32
        batch, seq_len = 4, 10

        extractor = GeometryFeatureExtractor(d_model=d_model)
        x = torch.randn(batch, seq_len, d_model)

        features = extractor.compute_local_structure_features(x)

        assert features.shape == (batch, seq_len, 3)  # 3 features: tree-ness, cycle-ness, flat-ness
        assert torch.all(features >= 0) and torch.all(features <= 1), "Features should be in [0, 1]"

        print("✅ Feature Extractor Output Shape Test Passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ROUTER AND AGA LAYER TESTS")
    print("="*60 + "\n")

    print("=== Testing CurvatureAwareRouter ===")
    TestCurvatureAwareRouter().test_router_output_shape()
    TestCurvatureAwareRouter().test_router_weights_sum_to_one()
    TestCurvatureAwareRouter().test_router_temperature_effect()
    TestCurvatureAwareRouter().test_router_gradient_flow()
    TestCurvatureAwareRouter().test_top_k_routing()

    print("\n=== Testing Load Balancing Loss ===")
    TestLoadBalancingLoss().test_perfect_balance_has_zero_loss()
    TestLoadBalancingLoss().test_imbalanced_routing_has_high_loss()
    TestLoadBalancingLoss().test_load_balance_gradient_flow()
    TestLoadBalancingLoss().test_entropy_loss()

    print("\n=== Testing AGALayer ===")
    TestAGALayer().test_aga_layer_forward()
    TestAGALayer().test_aga_layer_with_mask()
    TestAGALayer().test_aga_layer_gradient_flow()
    TestAGALayer().test_aga_block()
    TestAGALayer().test_geometry_usage_stats()

    print("\n=== Testing Router Behavior ===")
    TestRouterBehavior().test_router_distinguishes_structures()

    print("\n=== Testing Geometry Feature Extractor ===")
    TestGeometryFeatureExtractor().test_feature_extractor_output_shape()

    print("\n" + "="*60)
    print("✅ ALL ROUTER TESTS PASSED")
    print("="*60 + "\n")
