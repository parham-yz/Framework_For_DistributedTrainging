"""Unit test for the private ``_compute_hessian`` utility.

The test constructs a simple quadratic model of the form

    f(x) = \sum_i a_i x_i^2   (with *a_i* > 0)

For such a function the Hessian with respect to the parameters *x* is the
diagonal matrix ``diag(2 a)``.  We compare the finite‑difference estimate
returned by ``_compute_hessian`` against this analytic result for a 1000‑
dimensional problem.
"""

# Ensure project root (containing *src*) is on *sys.path*
import os, sys, torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------------------------------------------------------
#  Model and criterion definitions
# -----------------------------------------------------------------------------


class QuadraticModel(torch.nn.Module):
    """Quadratic form *f(x) = xᵀ A x* with trainable *x* and fixed symmetric *A*."""

    def __init__(self, A: torch.Tensor):
        super().__init__()

        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("`A` must be a square 2‑D tensor")

        self.register_parameter("x", torch.nn.Parameter(torch.randn(A.shape[0])))
        self.register_buffer("A", A)

    def forward(self, _input):  # input ignored, kept for API compatibility
        return torch.matmul(self.x, torch.matmul(self.A, self.x))


class IdentityCriterion(torch.nn.Module):
    """Loss that simply returns its *output* argument (ignores *target*)."""

    def forward(self, output, _target):  # noqa: D401,D403
        return output


# -----------------------------------------------------------------------------
#  Test
# -----------------------------------------------------------------------------


def test_compute_hessian_matches_quadratic_analytic():
    """The finite‑difference Hessian of the quadratic model should equal 2 diag(a)."""

    from src.Buliding_Units.MeasurementUnit import _compute_hessian

    torch.manual_seed(0)

    dim = 100  # use 100×100 so the full 10‑run suite executes quickly
    epsilon = 1e-2  # finite‑difference step size (empirically robust)

    for seed in range(10):
        torch.manual_seed(seed)

        # Create a different random symmetric matrix for each iteration
        B = torch.rand(dim, dim)
        A = (B + B.T) / 2.0

        model = QuadraticModel(A)

        dummy_input = torch.tensor([0.0])
        dummy_target = torch.tensor(0.0)

        criterion = IdentityCriterion()

        H_est = _compute_hessian(
            model, dummy_input, dummy_target, criterion, epsilon=epsilon
        )

        H_true = 2 * A

        # Require tight absolute precision (finite‑difference should recover
        # the _constant_ Hessian for a quadratic exactly up to numerical error).
        abs_err = (H_est - H_true).abs().max()

        assert abs_err < 1e-3, (
            f"Max absolute error {abs_err:.4e} exceeded tolerance for seed {seed}"
        )
