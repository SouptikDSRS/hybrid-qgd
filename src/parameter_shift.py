"""
parameter_shift.py
------------------
Gradient computation via the Parameter Shift Rule (PSR) for parameterised
quantum circuits.

Parameter Shift Rule
--------------------
For a parametrised quantum circuit f(θ):

    ∂f/∂θ_i = [ f(θ + π/2 · e_i) - f(θ - π/2 · e_i) ] / 2

where e_i is the unit vector in the i-th parameter direction.

This module is agnostic to the objective function implementation — it only
requires a callable that maps a parameter array to a scalar float loss.

For the hybrid QGD framework the callable wraps a parametrised Qiskit circuit,
but any differentiable (or non-differentiable) function can be used for
classical benchmarking.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Core PSR functions
# ─────────────────────────────────────────────────────────────────────────────

def parameter_shift_gradient(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute the gradient of `loss_fn` at `params` via the Parameter Shift Rule.

    Parameters
    ----------
    loss_fn : callable  params → float scalar loss
    params  : 1-D numpy array of current parameter values
    shift   : shift value (default π/2 for standard PSR)

    Returns
    -------
    gradients : numpy array of shape params.shape
    """
    params = np.asarray(params, dtype=float)
    n = len(params)
    grads = np.zeros(n)

    for i in range(n):
        # Forward shift
        p_plus = params.copy()
        p_plus[i] += shift
        f_plus = loss_fn(p_plus)

        # Backward shift
        p_minus = params.copy()
        p_minus[i] -= shift
        f_minus = loss_fn(p_minus)

        grads[i] = (f_plus - f_minus) / 2.0

    return grads


def parameter_shift_gradient_parallel(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """PSR gradient with all circuits batched for efficiency.

    Collects all 2n shifted parameter sets, evaluates them, then assembles
    the gradient. For simulators this is equivalent to the sequential version;
    for hardware with a job-batching API this reduces round-trips.

    Parameters
    ----------
    loss_fn : callable
    params  : 1-D parameter array
    shift   : PSR shift

    Returns
    -------
    gradients : numpy array
    """
    params = np.asarray(params, dtype=float)
    n = len(params)

    # Build all 2n shifted variants
    shifted_params = []
    for i in range(n):
        p_plus  = params.copy(); p_plus[i]  += shift
        p_minus = params.copy(); p_minus[i] -= shift
        shifted_params.append((i, '+', p_plus))
        shifted_params.append((i, '-', p_minus))

    # Evaluate (could be parallelised externally)
    results: dict[tuple, float] = {}
    for i, sign, p in shifted_params:
        results[(i, sign)] = loss_fn(p)

    # Assemble gradient
    grads = np.zeros(n)
    for i in range(n):
        grads[i] = (results[(i, '+')] - results[(i, '-')]) / 2.0

    return grads


# ─────────────────────────────────────────────────────────────────────────────
# Finite-difference gradient (classical baseline)
# ─────────────────────────────────────────────────────────────────────────────

def finite_difference_gradient(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference gradient — classical reference only.

    Parameters
    ----------
    loss_fn : callable
    params  : 1-D parameter array
    eps     : finite-difference step size

    Returns
    -------
    gradients : numpy array
    """
    params = np.asarray(params, dtype=float)
    n = len(params)
    grads = np.zeros(n)
    for i in range(n):
        p_plus  = params.copy(); p_plus[i]  += eps
        p_minus = params.copy(); p_minus[i] -= eps
        grads[i] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2.0 * eps)
    return grads


# ─────────────────────────────────────────────────────────────────────────────
# ParameterShiftEstimator class (stateful, tracks history)
# ─────────────────────────────────────────────────────────────────────────────

class ParameterShiftEstimator:
    """Stateful PSR gradient estimator.

    Wraps any loss function and accumulates gradient evaluation statistics
    (number of circuit evaluations, gradient norms, etc.).

    Parameters
    ----------
    loss_fn : callable  params → float
    shift   : PSR shift angle (default π/2)
    """

    def __init__(self,
                 loss_fn: Callable[[np.ndarray], float],
                 shift: float = np.pi / 2):
        self.loss_fn = loss_fn
        self.shift   = shift
        self._n_circuit_evals = 0
        self._gradient_history: list[np.ndarray] = []
        self._loss_history: list[float] = []

    # ── Gradient ──────────────────────────────────────────────────────────────

    def gradient(self, params: np.ndarray) -> np.ndarray:
        """Compute PSR gradient at params and record statistics.

        Each gradient call costs 2 * len(params) circuit evaluations.
        """
        params = np.asarray(params, dtype=float)
        n = len(params)
        grads = np.zeros(n)

        for i in range(n):
            p_plus  = params.copy(); p_plus[i]  += self.shift
            p_minus = params.copy(); p_minus[i] -= self.shift
            f_plus  = self.loss_fn(p_plus)
            f_minus = self.loss_fn(p_minus)
            grads[i] = (f_plus - f_minus) / 2.0
            self._n_circuit_evals += 2

        self._gradient_history.append(grads.copy())
        return grads

    def loss(self, params: np.ndarray) -> float:
        """Evaluate loss and record it."""
        val = self.loss_fn(np.asarray(params, dtype=float))
        self._loss_history.append(val)
        self._n_circuit_evals += 1
        return val

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def n_circuit_evals(self) -> int:
        return self._n_circuit_evals

    @property
    def gradient_norms(self) -> list[float]:
        return [float(np.linalg.norm(g)) for g in self._gradient_history]

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._n_circuit_evals = 0
        self._gradient_history = []
        self._loss_history = []
