"""
hybrid_optimizer.py
--------------------
Hybrid Quantum-Classical Gradient Descent Optimizer.

Algorithm
---------
For each iteration t:
  1.  Evaluate loss  L(x_t)  using objective function
  2.  Compute gradient  g_t  via Parameter Shift Rule
  3.  Scale step:  Δ_i = α · g_t[i]  (classical)
  4.  Update each parameter:  x_{t+1}[i] = x_t[i] - Δ_i
      using a QFT-based subtraction circuit
  5.  Record loss, gradient norm, and timing

Modes
-----
- "hybrid"    : QFT subtraction on quantum backend (default)
- "classical" : standard gradient descent (no quantum arithmetic)
- "debug"     : hybrid with verbose circuit output

The optimizer is fully configurable via config.yaml and works with any
backend (Aer simulator or IBM Runtime).
"""

from __future__ import annotations

import time
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Literal

from src.parameter_shift import ParameterShiftEstimator
from src.qft_arithmetic import QFTArithmetic
from src.utils import clip_to_range


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Stores the full trajectory of an optimization run."""
    final_params:   np.ndarray
    final_loss:     float
    loss_history:   list[float]          = field(default_factory=list)
    grad_norms:     list[float]          = field(default_factory=list)
    param_history:  list[np.ndarray]     = field(default_factory=list)
    n_iterations:   int                  = 0
    n_circuit_evals: int                 = 0
    converged:      bool                 = False
    mode:           str                  = "hybrid"
    wall_time_s:    float                = 0.0
    backend_name:   str                  = ""

    def as_dict(self) -> dict:
        return {
            "final_params":    self.final_params.tolist(),
            "final_loss":      self.final_loss,
            "loss_history":    self.loss_history,
            "grad_norms":      self.grad_norms,
            "n_iterations":    self.n_iterations,
            "n_circuit_evals": self.n_circuit_evals,
            "converged":       self.converged,
            "mode":            self.mode,
            "wall_time_s":     self.wall_time_s,
            "backend_name":    self.backend_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class HybridQFTOptimizer:
    """Hybrid quantum-classical gradient descent with QFT arithmetic.

    Parameters
    ----------
    loss_fn          : callable  params → float  (the objective)
    backend          : Qiskit backend for QFT circuit execution
    learning_rate    : step size α
    n_qubits         : QFT register width
    v_min, v_max     : representable value range for QFT encoding
    shots            : measurement shots per QFT circuit
    max_iterations   : maximum optimisation steps
    convergence_tol  : stop when |Δloss| < tol
    mode             : "hybrid" | "classical" | "debug"
    verbose          : print iteration summaries
    """

    def __init__(
        self,
        loss_fn:        Callable[[np.ndarray], float],
        backend=None,
        learning_rate:  float = 0.1,
        n_qubits:       int   = 4,
        v_min:          float = -2.0,
        v_max:          float = 2.0,
        shots:          int   = 4096,
        max_iterations: int   = 50,
        convergence_tol: float = 1e-6,
        mode:           Literal["hybrid", "classical", "debug"] = "hybrid",
        verbose:        bool  = True,
    ):
        self.loss_fn        = loss_fn
        self.backend        = backend
        self.lr             = learning_rate
        self.n_qubits       = n_qubits
        self.v_min          = v_min
        self.v_max          = v_max
        self.shots          = shots
        self.max_iter       = max_iterations
        self.tol            = convergence_tol
        self.mode           = mode
        self.verbose        = verbose

        # Sub-modules
        self._pse = ParameterShiftEstimator(loss_fn)
        self._qft: QFTArithmetic | None = None
        if mode in ("hybrid", "debug"):
            if backend is None:
                raise ValueError("backend must be provided for hybrid mode")
            self._qft = QFTArithmetic(n_qubits=n_qubits, v_min=v_min, v_max=v_max)

    # ── Parameter update ──────────────────────────────────────────────────────

    def _update_param_hybrid(self, x_i: float, delta_i: float) -> float:
        """Update a single parameter using QFT subtraction circuit.

        x_{t+1}[i] = x_t[i] - α·g_t[i]

        Both values are clipped to [v_min, v_max] before encoding.
        """
        x_clipped = clip_to_range(x_i, self.v_min, self.v_max)
        d_clipped  = clip_to_range(delta_i, self.v_min, self.v_max)

        result = self._qft.subtract(x_clipped, d_clipped,
                                    self.backend, shots=self.shots)

        if self.mode == "debug":
            classical = x_i - delta_i
            print(f"    QFT: {x_clipped:.4f} - {d_clipped:.4f} = {result:.4f}"
                  f"  (classical: {classical:.4f},"
                  f"  err: {abs(result-classical):.4f})")

        return result

    def _update_param_classical(self, x_i: float, delta_i: float) -> float:
        """Standard classical parameter update (baseline)."""
        return x_i - delta_i

    # ── Optimization loop ─────────────────────────────────────────────────────

    def optimize(self, initial_params: np.ndarray) -> OptimizationResult:
        """Run the hybrid optimization loop.

        Parameters
        ----------
        initial_params : starting parameter values

        Returns
        -------
        OptimizationResult with full trajectory
        """
        params = np.asarray(initial_params, dtype=float).copy()
        n_params = len(params)

        loss_history:  list[float]      = []
        grad_norms:    list[float]      = []
        param_history: list[np.ndarray] = []

        converged  = False
        t_start    = time.time()
        prev_loss  = None

        backend_name = getattr(self.backend, "name", str(self.backend))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  HybridQFTOptimizer — mode={self.mode}")
            print(f"  Backend: {backend_name}")
            print(f"  n_params={n_params}, lr={self.lr}, "
                  f"n_qubits={self.n_qubits}, shots={self.shots}")
            print(f"{'='*60}")

        for t in range(self.max_iter):
            # ── 1. Evaluate loss ──
            loss = self._pse.loss(params)
            loss_history.append(loss)
            param_history.append(params.copy())

            # ── 2. Convergence check ──
            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                converged = True
                if self.verbose:
                    print(f"  [iter {t:3d}] Converged (Δloss < {self.tol:.1e})")
                break
            prev_loss = loss

            # ── 3. Compute gradient via PSR ──
            grads = self._pse.gradient(params)
            gnorm = float(np.linalg.norm(grads))
            grad_norms.append(gnorm)

            if self.verbose:
                print(f"  [iter {t:3d}] loss={loss:.6f}  |∇|={gnorm:.4f}"
                      f"  params={np.round(params, 4)}")

            # ── 4. Compute scaled step ──
            deltas = self.lr * grads

            # ── 5. Update parameters ──
            new_params = np.zeros_like(params)
            for i in range(n_params):
                if self.mode in ("hybrid", "debug"):
                    new_params[i] = self._update_param_hybrid(params[i], deltas[i])
                else:
                    new_params[i] = self._update_param_classical(params[i], deltas[i])

            params = new_params

        # Final loss
        final_loss = self._pse.loss(params)
        loss_history.append(final_loss)
        param_history.append(params.copy())
        wall_time = time.time() - t_start

        if self.verbose:
            print(f"\n  Final loss : {final_loss:.6f}")
            print(f"  Wall time  : {wall_time:.2f}s")
            print(f"  Iterations : {len(loss_history)-1}")
            print(f"  Converged  : {converged}")

        return OptimizationResult(
            final_params   = params,
            final_loss     = final_loss,
            loss_history   = loss_history,
            grad_norms     = grad_norms,
            param_history  = param_history,
            n_iterations   = len(loss_history) - 1,
            n_circuit_evals= self._pse.n_circuit_evals,
            converged      = converged,
            mode           = self.mode,
            wall_time_s    = wall_time,
            backend_name   = backend_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Classical GD baseline (no quantum, for comparison)
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalGradientDescent:
    """Standard gradient descent baseline.

    Uses finite-difference gradient by default (or analytic if provided).

    Parameters
    ----------
    loss_fn       : callable
    gradient_fn   : optional analytic gradient function
    learning_rate : step size
    max_iterations: iteration limit
    convergence_tol: stopping tolerance
    verbose       : print progress
    """

    def __init__(
        self,
        loss_fn:        Callable[[np.ndarray], float],
        gradient_fn:    Callable[[np.ndarray], np.ndarray] | None = None,
        learning_rate:  float = 0.1,
        max_iterations: int   = 50,
        convergence_tol: float = 1e-6,
        verbose:        bool  = True,
    ):
        self.loss_fn    = loss_fn
        self.grad_fn    = gradient_fn
        self.lr         = learning_rate
        self.max_iter   = max_iterations
        self.tol        = convergence_tol
        self.verbose    = verbose

    def optimize(self, initial_params: np.ndarray) -> OptimizationResult:
        params = np.asarray(initial_params, dtype=float).copy()

        loss_history:  list[float]      = []
        grad_norms:    list[float]      = []
        param_history: list[np.ndarray] = []

        converged  = False
        t_start    = time.time()
        prev_loss  = None

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  ClassicalGradientDescent  lr={self.lr}")
            print(f"{'='*60}")

        for t in range(self.max_iter):
            loss = self.loss_fn(params)
            loss_history.append(loss)
            param_history.append(params.copy())

            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                converged = True
                break
            prev_loss = loss

            if self.grad_fn is not None:
                grads = self.grad_fn(params)
            else:
                from src.parameter_shift import finite_difference_gradient
                grads = finite_difference_gradient(self.loss_fn, params)

            gnorm = float(np.linalg.norm(grads))
            grad_norms.append(gnorm)

            if self.verbose:
                print(f"  [iter {t:3d}] loss={loss:.6f}  |∇|={gnorm:.4f}")

            params = params - self.lr * grads

        final_loss = self.loss_fn(params)
        loss_history.append(final_loss)
        param_history.append(params.copy())
        wall_time = time.time() - t_start

        return OptimizationResult(
            final_params   = params,
            final_loss     = final_loss,
            loss_history   = loss_history,
            grad_norms     = grad_norms,
            param_history  = param_history,
            n_iterations   = len(loss_history) - 1,
            n_circuit_evals= 0,
            converged      = converged,
            mode           = "classical",
            wall_time_s    = wall_time,
            backend_name   = "cpu",
        )
