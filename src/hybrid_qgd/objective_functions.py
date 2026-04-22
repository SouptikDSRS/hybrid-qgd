"""
objective_functions.py
-----------------------
A library of differentiable test objective functions for benchmarking
the hybrid QFT gradient descent optimizer.

All functions accept a numpy parameter array and return a scalar float.
They also expose their analytic gradient for comparison.

Available functions
-------------------
- quadratic      : f(x) = Σ x_i²                  (global min at 0)
- rosenbrock     : f(x,y) = (1-x)² + 100(y-x²)²   (global min at (1,1))
- himmelblau     : f(x,y) = (x²+y-11)²+(x+y²-7)²  (4 global minima)
- sin_cos        : f(x,y) = sin(x)·cos(y)          (many local minima)
- sphere         : f(x) = Σ x_i²                   (alias of quadratic)

Quantum Loss Functions
-----------------------
- variational_energy: expectation value of a PauliZ observable via
  a parameterised Ry circuit. This is the loss used in VQE-style
  experiments and demonstrates PSR in its natural context.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Classical analytic functions
# ─────────────────────────────────────────────────────────────────────────────

def quadratic(params: np.ndarray) -> float:
    """f(x) = Σ x_i²  — convex, global minimum at origin."""
    return float(np.sum(np.asarray(params) ** 2))


def quadratic_gradient(params: np.ndarray) -> np.ndarray:
    """Analytic gradient of quadratic: 2x."""
    return 2.0 * np.asarray(params, dtype=float)


def rosenbrock(params: np.ndarray) -> float:
    """Rosenbrock banana function (2D).

    f(x, y) = (1 - x)² + 100·(y - x²)²
    Global minimum at (1, 1) with f=0.
    """
    params = np.asarray(params, dtype=float)
    x, y = params[0], params[1]
    return float((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)


def rosenbrock_gradient(params: np.ndarray) -> np.ndarray:
    """Analytic gradient of Rosenbrock."""
    params = np.asarray(params, dtype=float)
    x, y = params[0], params[1]
    dfdx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dfdy = 200 * (y - x ** 2)
    return np.array([dfdx, dfdy])


def himmelblau(params: np.ndarray) -> float:
    """Himmelblau's function (2D).

    f(x, y) = (x² + y - 11)² + (x + y² - 7)²
    Four global minima, all with f=0.
    """
    params = np.asarray(params, dtype=float)
    x, y = params[0], params[1]
    return float((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2)


def himmelblau_gradient(params: np.ndarray) -> np.ndarray:
    """Analytic gradient of Himmelblau."""
    params = np.asarray(params, dtype=float)
    x, y = params[0], params[1]
    dfdx = 2 * (x ** 2 + y - 11) * 2 * x + 2 * (x + y ** 2 - 7)
    dfdy = 2 * (x ** 2 + y - 11)          + 2 * (x + y ** 2 - 7) * 2 * y
    return np.array([dfdx, dfdy])


def sin_cos(params: np.ndarray) -> float:
    """f(x, y) = -sin(x)·cos(y) — non-convex, many local optima."""
    params = np.asarray(params, dtype=float)
    x, y = params[0], params[1]
    return float(-np.sin(x) * np.cos(y))


def sin_cos_gradient(params: np.ndarray) -> np.ndarray:
    """Analytic gradient of sin_cos."""
    params = np.asarray(params, dtype=float)
    x, y = params[0], params[1]
    dfdx = -np.cos(x) * np.cos(y)
    dfdy =  np.sin(x) * np.sin(y)
    return np.array([dfdx, dfdy])


def sphere(params: np.ndarray) -> float:
    """Sphere function (alias of quadratic, supports any dimension)."""
    return quadratic(params)


sphere_gradient = quadratic_gradient


# ─────────────────────────────────────────────────────────────────────────────
# Quantum loss function — variational energy
# ─────────────────────────────────────────────────────────────────────────────

class VariationalEnergyLoss:
    """Loss function based on the expectation value ⟨ψ(θ)|Z|ψ(θ)⟩.

    Circuit: Ry(θ_i) on each qubit, then measure PauliZ expectation.

    This is a canonical VQE-style loss that is natively differentiable via PSR.
    Runs on a provided Qiskit backend.

    Parameters
    ----------
    n_qubits : number of qubits (= number of parameters)
    backend  : Qiskit backend (Aer or IBM Runtime)
    shots    : measurement shots per evaluation
    """

    def __init__(self, n_qubits: int, backend, shots: int = 2048):
        self.n_qubits = n_qubits
        self.backend  = backend
        self.shots    = shots
        self._build_estimator()

    def _build_estimator(self):
        """Set up Qiskit Estimator primitive."""
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator
        try:
            if isinstance(self.backend, AerSimulator):
                self._estimator = AerEstimator()
            else:
                from qiskit_ibm_runtime import EstimatorV2
                self._estimator = EstimatorV2(self.backend)
        except Exception:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator
            self._estimator = AerEstimator()

    def _build_circuit(self, params: np.ndarray):
        """Ry(θ_i) circuit with SparsePauliOp observable."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        n = self.n_qubits
        pv = ParameterVector("θ", length=n)
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.ry(pv[i], i)

        # Observable: sum of Z on each qubit
        obs = SparsePauliOp.from_list([
            ("I" * (n - 1 - i) + "Z" + "I" * i, 1.0)
            for i in range(n)
        ])
        return qc, obs, dict(zip(pv, params))

    def __call__(self, params: np.ndarray) -> float:
        """Evaluate ⟨Z⟩ expectation value."""
        params = np.asarray(params, dtype=float)
        qc, obs, param_dict = self._build_circuit(params)
        bound = qc.assign_parameters(param_dict)

        try:
            job = self._estimator.run([(bound, obs)])
            result = job.result()
            # EstimatorV2 PrimitiveResult
            expval = float(result[0].data.evs)
        except Exception as e:
            # Fallback: statevector calculation
            from qiskit.quantum_info import Statevector
            sv = Statevector(bound)
            expval = float(sv.expectation_value(obs).real)

        return expval


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

FUNCTIONS: dict[str, Callable] = {
    "quadratic":  quadratic,
    "rosenbrock": rosenbrock,
    "himmelblau": himmelblau,
    "sin_cos":    sin_cos,
    "sphere":     sphere,
}

GRADIENTS: dict[str, Callable] = {
    "quadratic":  quadratic_gradient,
    "rosenbrock": rosenbrock_gradient,
    "himmelblau": himmelblau_gradient,
    "sin_cos":    sin_cos_gradient,
    "sphere":     sphere_gradient,
}

OPTIMA: dict[str, tuple] = {
    "quadratic":  (np.zeros(2), 0.0),
    "rosenbrock": (np.array([1.0, 1.0]), 0.0),
    "sphere":     (np.zeros(2), 0.0),
}


def get_function(name: str) -> Callable:
    """Return objective function by name."""
    if name not in FUNCTIONS:
        raise ValueError(f"Unknown function '{name}'. "
                         f"Choose from {list(FUNCTIONS.keys())}")
    return FUNCTIONS[name]


def get_gradient(name: str) -> Callable:
    """Return analytic gradient by name."""
    if name not in GRADIENTS:
        raise ValueError(f"Unknown gradient '{name}'.")
    return GRADIENTS[name]
