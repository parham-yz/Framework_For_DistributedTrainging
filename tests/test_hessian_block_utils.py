"""Test end‑to‑end block handling utilities used in Hessian measurement.

This verifies that the sequence

    1. _compute_hessian
    2. _get_block_parameter_counts
    3. _decompose_matrix_to_blocks
    4. _compute_block_operator_norms

produces operator norms that match analytic values for a quadratic model with
known Hessian.
"""

import os
import sys

import torch


# -----------------------------------------------------------------------------
#  Make *src* importable when running via ``pytest`` from repository root.
# -----------------------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.Buliding_Units.MeasurementUnit import (
    _compute_hessian,
    _get_block_parameter_counts,
    _decompose_matrix_to_blocks,
    _compute_block_operator_norms,
)


# -----------------------------------------------------------------------------
#  Helper model
# -----------------------------------------------------------------------------


class QuadBlock(torch.nn.Module):
    """Block containing a parameter vector of given size."""

    def __init__(self, size: int):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(size))

    def forward(self):  # not used; parameters accessed from parent model
        raise RuntimeError("QuadBlock should not be called directly")


class QuadraticBlockModel(torch.nn.Module):
    """Quadratic form model *f(x) = xᵀ A x* with parameters split into blocks."""

    def __init__(self, A: torch.Tensor, block_sizes):
        super().__init__()

        if sum(block_sizes) != A.shape[0]:
            raise ValueError("Sum of block sizes must equal dimension of A")

        # Register the symmetric matrix A as a buffer (non‑trainable)
        self.register_buffer("A", A)

        # Create blocks
        self.blocks = torch.nn.ModuleList([QuadBlock(sz) for sz in block_sizes])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, _input):  # input is ignored (API compatibility)
        x_parts = [blk.x for blk in self.blocks]
        x = torch.cat(x_parts)
        return torch.matmul(x, torch.matmul(self.A, x))


class IdentityCriterion(torch.nn.Module):
    def forward(self, output, _target):
        return output


# -----------------------------------------------------------------------------
#  Test
# -----------------------------------------------------------------------------


def test_block_operator_norm_pipeline():
    dim = 30  # small enough for speed yet exercises block logic
    torch.manual_seed(42)

    # Define arbitrary (but deterministic) block sizes that sum to *dim*
    block_sizes = [10, 8, 12]

    for seed in range(5):
        torch.manual_seed(seed)

        # Random symmetric positive‑semi‑definite A (not required PSD but keeps norms reasonable)
        B = torch.rand(dim, dim)
        A = (B + B.T) / 2.0

        model = QuadraticBlockModel(A, block_sizes)

        dummy_input = torch.tensor([0.0])
        criterion = IdentityCriterion()

        # Finite difference Hessian
        H_est = _compute_hessian(
            model, dummy_input, torch.tensor(0.0), criterion, epsilon=1e-2
        )

        # --- Pipeline under test ------------------------------------------------
        ls = _get_block_parameter_counts(model)
        H_blocks = _decompose_matrix_to_blocks(H_est, ls)
        block_norms_est = _compute_block_operator_norms(H_blocks)

        # --- Analytic reference -------------------------------------------------
        H_true = 2 * A

        # Decompose analytic Hessian manually into blocks
        idx = 0
        true_blocks = []
        for sz_i in ls:
            row_blocks = []
            jdx = 0
            for sz_j in ls:
                row_blocks.append(H_true[idx : idx + sz_i, jdx : jdx + sz_j])
                jdx += sz_j
            true_blocks.append(row_blocks)
            idx += sz_i

        block_norms_true = _compute_block_operator_norms(true_blocks)

        # ----------------------------------------------------------------------
        # Assertions
        # ----------------------------------------------------------------------

        # The estimated block norms should match the analytic values very
        # closely. Allow an absolute deviation of 1e‑3.
        assert torch.allclose(block_norms_est, block_norms_true, atol=1e-3, rtol=1e-3), (
            f"Block norm mismatch for seed {seed}"
        )
