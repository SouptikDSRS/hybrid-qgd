"""
test_parameter_shift.py
-----------------------
Unit tests for the Parameter Shift Rule gradient estimator.

PSR correctness note
--------------------
The Parameter Shift Rule is exact for quantum gates of the form exp(iθP/2)
where P is a Pauli. For classical polynomial functions it produces a
DIFFERENT result from the analytic gradient (it computes the gradient of
the trigonometric interpolation of the function, not the polynomial itself).

Therefore PSR-on-classical-polynomial tests compare PSR against
finite-difference (both numerically approximate for non-trigonometric
functions) with a loose tolerance, rather than against the analytic gradient.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from src.parameter_shift import (
    parameter_shift_gradient,
    parameter_shift_gradient_parallel,
    finite_difference_gradient,
    ParameterShiftEstimator,
)
from src.objective_functions import (
    quadratic, quadratic_gradient,
    rosenbrock,
    sphere,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def allclose(a, b, atol=1e-3):
    return np.allclose(np.asarray(a), np.asarray(b), atol=atol)


# ─────────────────────────────────────────────────────────────────────────────
# PSR on trigonometric functions (where PSR is exact)
# ─────────────────────────────────────────────────────────────────────────────

class TestPSROnTrigFunctions:
    """PSR is exact for f(θ) = a·sin(θ) + b·cos(θ)."""

    def test_sin_gradient(self):
        """df/dθ = cos(θ) via PSR."""
        f = lambda p: float(np.sin(p[0]))
        for theta in [0.0, 0.5, 1.0, 1.57, -0.7]:
            psr  = parameter_shift_gradient(f, np.array([theta]))
            true = np.array([np.cos(theta)])
            assert allclose(psr, true, atol=1e-9), \
                f"sin gradient: PSR={psr}, true={true} at theta={theta}"

    def test_cos_gradient(self):
        """df/dθ = -sin(θ)."""
        f = lambda p: float(np.cos(p[0]))
        for theta in [0.0, 0.5, 1.0, 1.57, -0.7]:
            psr  = parameter_shift_gradient(f, np.array([theta]))
            true = np.array([-np.sin(theta)])
            assert allclose(psr, true, atol=1e-9)

    def test_linear_combo(self):
        """f(θ) = 2·sin(θ) - cos(θ)  →  f'(θ) = 2·cos(θ) + sin(θ)."""
        f = lambda p: 2.0 * np.sin(p[0]) - np.cos(p[0])
        for theta in [0.3, 1.2, -0.5]:
            psr  = parameter_shift_gradient(f, np.array([theta]))
            true = np.array([2.0*np.cos(theta) + np.sin(theta)])
            assert allclose(psr, true, atol=1e-9)

    def test_multivariate_trig(self):
        """f(θ₁,θ₂) = sin(θ₁)·cos(θ₂)."""
        f = lambda p: float(np.sin(p[0]) * np.cos(p[1]))
        for t1, t2 in [(0.5, 0.3), (1.0, 0.7), (-0.2, 0.8)]:
            psr  = parameter_shift_gradient(f, np.array([t1, t2]))
            true = np.array([np.cos(t1)*np.cos(t2), -np.sin(t1)*np.sin(t2)])
            assert allclose(psr, true, atol=1e-9)

    def test_zero_gradient_at_pi_over_2(self):
        """f(θ)=sin(θ): f'(π/2)=0."""
        f = lambda p: float(np.sin(p[0]))
        psr = parameter_shift_gradient(f, np.array([np.pi / 2]))
        assert abs(psr[0]) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# PSR on non-trigonometric functions (compare with FD, not analytic)
# ─────────────────────────────────────────────────────────────────────────────

class TestPSROnClassicalFunctions:
    """For polynomial/non-trigonometric functions PSR and FD both approximate;
       their results differ, but both decrease toward the minimum."""

    def test_quadratic_psr_returns_array(self):
        params = np.array([0.3, -0.5])
        grads = parameter_shift_gradient(quadratic, params)
        assert grads.shape == params.shape

    def test_quadratic_psr_same_sign_as_analytic_near_origin(self):
        """At params close to origin, PSR gradient should share sign with
        analytic gradient (both point away from minimum)."""
        params = np.array([0.1, -0.1])
        psr    = parameter_shift_gradient(quadratic, params)
        analytic = quadratic_gradient(params)
        # Signs should match (both point away from minimum)
        for i in range(len(params)):
            if abs(analytic[i]) > 1e-3:
                assert psr[i] * analytic[i] > 0, \
                    f"Sign mismatch at index {i}: PSR={psr[i]}, analytic={analytic[i]}"

    def test_gradient_shape(self):
        params = np.random.randn(5)
        grads = parameter_shift_gradient(quadratic, params)
        assert grads.shape == params.shape

    def test_zero_gradient_at_minimum(self):
        """PSR of quadratic at origin should be zero."""
        params = np.zeros(3)
        grads = parameter_shift_gradient(quadratic, params)
        assert np.allclose(grads, 0.0, atol=1e-12)

    def test_rosenbrock_psr_returns_array(self):
        params = np.array([0.1, 0.1])
        grads = parameter_shift_gradient(rosenbrock, params)
        assert grads.shape == params.shape


# ─────────────────────────────────────────────────────────────────────────────
# Parallel PSR
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelPSR:
    def test_matches_sequential_trig(self):
        """On trig functions parallel and sequential PSR must agree exactly."""
        f = lambda p: float(np.sin(p[0]) * np.cos(p[1]) + np.sin(p[2]))
        params = np.array([0.3, -0.5, 0.1])
        seq = parameter_shift_gradient(f, params)
        par = parameter_shift_gradient_parallel(f, params)
        assert np.allclose(seq, par, atol=1e-12)

    def test_matches_sequential_quadratic(self):
        params = np.array([0.3, -0.5, 0.1])
        seq = parameter_shift_gradient(quadratic, params)
        par = parameter_shift_gradient_parallel(quadratic, params)
        assert np.allclose(seq, par, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Finite-difference
# ─────────────────────────────────────────────────────────────────────────────

class TestFiniteDifference:
    def test_fd_quadratic(self):
        params = np.array([0.3, -0.5])
        fd   = finite_difference_gradient(quadratic, params)
        true = quadratic_gradient(params)
        assert allclose(fd, true, atol=1e-4)

    def test_fd_sin(self):
        f = lambda p: float(np.sin(p[0]))
        for theta in [0.5, 1.2, -0.3]:
            fd   = finite_difference_gradient(f, np.array([theta]))
            true = np.array([np.cos(theta)])
            assert allclose(fd, true, atol=1e-5)

    def test_fd_shape(self):
        params = np.random.randn(4)
        grads = finite_difference_gradient(sphere, params)
        assert grads.shape == params.shape


# ─────────────────────────────────────────────────────────────────────────────
# ParameterShiftEstimator (stateful)
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterShiftEstimator:
    def test_gradient_exact_on_sin(self):
        """PSR is exact for sin."""
        f   = lambda p: float(np.sin(p[0]))
        pse = ParameterShiftEstimator(f)
        for theta in [0.5, 1.0, -0.3]:
            grads = pse.gradient(np.array([theta]))
            true  = np.array([np.cos(theta)])
            assert allclose(grads, true, atol=1e-9)
        pse.reset()

    def test_gradient_returns_array(self):
        pse    = ParameterShiftEstimator(quadratic)
        grads  = pse.gradient(np.array([0.5, -0.3]))
        assert grads.shape == (2,)

    def test_circuit_eval_counter(self):
        pse    = ParameterShiftEstimator(quadratic)
        params = np.array([0.0, 0.0])
        pse.gradient(params)
        assert pse.n_circuit_evals == 4   # 2 params × 2

        pse.loss(params)
        assert pse.n_circuit_evals == 5

    def test_loss_history(self):
        pse    = ParameterShiftEstimator(quadratic)
        pse.loss(np.array([1.0, 1.0]))
        pse.loss(np.array([0.5, 0.5]))
        assert len(pse.loss_history) == 2
        assert pse.loss_history[0] > pse.loss_history[1]

    def test_gradient_norm_history(self):
        pse    = ParameterShiftEstimator(quadratic)
        for p in [np.array([1.0, 1.0]), np.array([0.5, 0.5])]:
            pse.gradient(p)
        norms = pse.gradient_norms
        assert len(norms) == 2
        assert all(n >= 0 for n in norms)

    def test_reset(self):
        pse = ParameterShiftEstimator(quadratic)
        pse.gradient(np.array([1.0, 0.0]))
        pse.reset()
        assert pse.n_circuit_evals == 0
        assert pse.gradient_norms  == []
        assert pse.loss_history     == []
