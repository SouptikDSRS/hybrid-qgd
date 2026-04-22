"""
test_hybrid_optimizer.py
------------------------
Integration tests for HybridQFTOptimizer and ClassicalGradientDescent.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from qiskit_aer import AerSimulator

from src.hybrid_optimizer import HybridQFTOptimizer, ClassicalGradientDescent, OptimizationResult
from src.objective_functions import quadratic, quadratic_gradient


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backend():
    return AerSimulator()


@pytest.fixture(scope="module")
def init_params():
    return np.array([0.5, -0.5])


# ─────────────────────────────────────────────────────────────────────────────
# OptimizationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizationResult:
    def test_as_dict_keys(self):
        r = OptimizationResult(
            final_params=np.zeros(2),
            final_loss=0.0,
        )
        d = r.as_dict()
        for key in ["final_params", "final_loss", "loss_history",
                    "n_iterations", "converged", "mode"]:
            assert key in d

    def test_final_params_serializable(self):
        r = OptimizationResult(final_params=np.array([1.0, 2.0]), final_loss=1.5)
        d = r.as_dict()
        assert isinstance(d["final_params"], list)


# ─────────────────────────────────────────────────────────────────────────────
# ClassicalGradientDescent
# ─────────────────────────────────────────────────────────────────────────────

class TestClassicalGD:
    def test_runs_without_error(self, init_params):
        cgd = ClassicalGradientDescent(
            loss_fn=quadratic,
            gradient_fn=quadratic_gradient,
            learning_rate=0.1,
            max_iterations=20,
            verbose=False,
        )
        result = cgd.optimize(init_params.copy())
        assert isinstance(result, OptimizationResult)

    def test_loss_decreases(self, init_params):
        cgd = ClassicalGradientDescent(
            loss_fn=quadratic,
            gradient_fn=quadratic_gradient,
            learning_rate=0.1,
            max_iterations=30,
            verbose=False,
        )
        result = cgd.optimize(init_params.copy())
        assert result.final_loss < quadratic(init_params), \
            f"Loss did not decrease: {result.final_loss} >= {quadratic(init_params)}"

    def test_converges_to_minimum(self, init_params):
        cgd = ClassicalGradientDescent(
            loss_fn=quadratic,
            gradient_fn=quadratic_gradient,
            learning_rate=0.2,
            max_iterations=200,
            convergence_tol=1e-8,
            verbose=False,
        )
        result = cgd.optimize(init_params.copy())
        assert result.final_loss < 0.01

    def test_result_mode_is_classical(self, init_params):
        cgd = ClassicalGradientDescent(
            loss_fn=quadratic,
            gradient_fn=quadratic_gradient,
            learning_rate=0.1,
            max_iterations=5,
            verbose=False,
        )
        result = cgd.optimize(init_params.copy())
        assert result.mode == "classical"

    def test_loss_history_length(self, init_params):
        n_iter = 10
        cgd = ClassicalGradientDescent(
            loss_fn=quadratic,
            gradient_fn=quadratic_gradient,
            learning_rate=0.1,
            max_iterations=n_iter,
            convergence_tol=1e-20,   # prevent early stop
            verbose=False,
        )
        result = cgd.optimize(init_params.copy())
        assert len(result.loss_history) == n_iter + 1  # +1 for final


# ─────────────────────────────────────────────────────────────────────────────
# HybridQFTOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridOptimizer:
    def test_runs_hybrid_mode(self, backend, init_params):
        opt = HybridQFTOptimizer(
            loss_fn=quadratic,
            backend=backend,
            learning_rate=0.1,
            n_qubits=4,
            v_min=-2.0,
            v_max=2.0,
            shots=1024,
            max_iterations=5,
            verbose=False,
        )
        result = opt.optimize(init_params.copy())
        assert isinstance(result, OptimizationResult)
        assert result.mode == "hybrid"

    def test_loss_history_not_empty(self, backend, init_params):
        opt = HybridQFTOptimizer(
            loss_fn=quadratic,
            backend=backend,
            learning_rate=0.1,
            n_qubits=4,
            v_min=-2.0,
            v_max=2.0,
            shots=1024,
            max_iterations=5,
            verbose=False,
        )
        result = opt.optimize(init_params.copy())
        assert len(result.loss_history) > 0

    def test_hybrid_requires_backend(self):
        with pytest.raises(ValueError, match="backend"):
            HybridQFTOptimizer(
                loss_fn=quadratic,
                backend=None,
                learning_rate=0.1,
                mode="hybrid",
            )

    def test_classical_mode_no_backend_needed(self, init_params):
        """Classical mode should work even without a backend."""
        opt = HybridQFTOptimizer(
            loss_fn=quadratic,
            backend=None,
            learning_rate=0.1,
            max_iterations=5,
            mode="classical",
            verbose=False,
        )
        result = opt.optimize(init_params.copy())
        assert result.mode == "classical"

    def test_circuit_evals_positive(self, backend, init_params):
        opt = HybridQFTOptimizer(
            loss_fn=quadratic,
            backend=backend,
            learning_rate=0.1,
            n_qubits=4,
            v_min=-2.0,
            v_max=2.0,
            shots=512,
            max_iterations=3,
            verbose=False,
        )
        result = opt.optimize(init_params.copy())
        assert result.n_circuit_evals > 0

    def test_wall_time_positive(self, backend, init_params):
        opt = HybridQFTOptimizer(
            loss_fn=quadratic,
            backend=backend,
            learning_rate=0.1,
            n_qubits=4,
            v_min=-2.0,
            v_max=2.0,
            shots=512,
            max_iterations=3,
            verbose=False,
        )
        result = opt.optimize(init_params.copy())
        assert result.wall_time_s > 0
