"""
noise_model.py
--------------
Constructs parametric noise models for Aer simulation.

Supported noise channels
-------------------------
- Depolarizing noise on single-qubit and two-qubit gates
- Thermal relaxation (T1/T2) noise
- Readout (measurement) error
- Combined realistic noise model

These models are used in Experiment 4 (noise sensitivity analysis).
"""

from __future__ import annotations

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_depolarizing_model(
    single_qubit_error: float,
    two_qubit_error:    float | None = None,
) -> NoiseModel:
    """Depolarizing noise model.

    Applies a depolarizing channel after every 1Q and 2Q gate.

    Parameters
    ----------
    single_qubit_error : error rate for 1Q gates (e.g. 0.001)
    two_qubit_error    : error rate for 2Q gates; defaults to 10× 1Q error

    Returns
    -------
    NoiseModel
    """
    if two_qubit_error is None:
        two_qubit_error = 10 * single_qubit_error

    noise = NoiseModel()

    # Single-qubit gates
    sq_error = depolarizing_error(single_qubit_error, 1)
    for gate in ["u", "u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z",
                 "s", "sdg", "t", "tdg", "p"]:
        noise.add_all_qubit_quantum_error(sq_error, gate)

    # Two-qubit gates
    tq_error = depolarizing_error(two_qubit_error, 2)
    for gate in ["cx", "cz", "cp", "swap"]:
        noise.add_all_qubit_quantum_error(tq_error, gate)

    return noise


def build_thermal_relaxation_model(
    t1_us:      float = 100.0,
    t2_us:      float = 80.0,
    gate_time_ns: float = 50.0,
) -> NoiseModel:
    """Thermal relaxation (T1/T2) noise model.

    Parameters
    ----------
    t1_us        : T1 relaxation time in microseconds
    t2_us        : T2 dephasing time in microseconds (must be ≤ 2·T1)
    gate_time_ns : gate duration in nanoseconds

    Returns
    -------
    NoiseModel
    """
    t2_us = min(t2_us, 2 * t1_us)  # enforce T2 ≤ 2T1

    t1_ns = t1_us * 1e3
    t2_ns = t2_us * 1e3

    noise = NoiseModel()
    relaxation = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns)

    for gate in ["u", "u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "p"]:
        noise.add_all_qubit_quantum_error(relaxation, gate)

    # 2Q gate (longer duration)
    relaxation_2q = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns * 10)
    for gate in ["cx", "cz", "cp"]:
        noise.add_all_qubit_quantum_error(
            relaxation_2q.expand(relaxation_2q), gate
        )

    return noise


def build_readout_error_model(readout_error: float = 0.01) -> NoiseModel:
    """Readout (measurement) error model.

    Adds a symmetric bitflip readout error to all qubits.

    Parameters
    ----------
    readout_error : probability of flipping 0→1 or 1→0

    Returns
    -------
    NoiseModel
    """
    noise = NoiseModel()
    p0g1 = readout_error  # prob 0 given prepared 1
    p1g0 = readout_error  # prob 1 given prepared 0
    ro_error = ReadoutError([[1 - p1g0, p1g0], [p0g1, 1 - p0g1]])
    noise.add_all_qubit_readout_error(ro_error)
    return noise


def build_combined_noise_model(
    depolarizing_rate: float = 0.001,
    readout_error:     float = 0.01,
    t1_us:             float = 100.0,
    t2_us:             float = 80.0,
) -> NoiseModel:
    """Combined realistic noise model for NISQ simulation.

    Combines depolarizing + readout errors. Thermal relaxation is not
    combined because it requires matching gate times carefully; use
    build_thermal_relaxation_model separately for T1/T2 studies.

    Parameters
    ----------
    depolarizing_rate : single-qubit gate error
    readout_error     : measurement bit-flip probability
    t1_us, t2_us      : relaxation times (not applied in this model)

    Returns
    -------
    NoiseModel
    """
    noise = NoiseModel()

    # Depolarizing on gates
    sq_error = depolarizing_error(depolarizing_rate, 1)
    tq_error = depolarizing_error(10 * depolarizing_rate, 2)

    for gate in ["u", "u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "p"]:
        noise.add_all_qubit_quantum_error(sq_error, gate)
    for gate in ["cx", "cz", "cp"]:
        noise.add_all_qubit_quantum_error(tq_error, gate)

    # Readout
    ro_error = ReadoutError([
        [1 - readout_error, readout_error],
        [readout_error, 1 - readout_error],
    ])
    noise.add_all_qubit_readout_error(ro_error)

    return noise


def noisy_aer_backend(depolarizing_rate: float = 0.001,
                      readout_error: float = 0.01):
    """Return an AerSimulator configured with a combined noise model.

    Convenience factory for experiments.

    Parameters
    ----------
    depolarizing_rate : gate error rate
    readout_error     : readout error rate

    Returns
    -------
    AerSimulator with noise model attached
    """
    from qiskit_aer import AerSimulator

    nm = build_combined_noise_model(
        depolarizing_rate=depolarizing_rate,
        readout_error=readout_error,
    )
    return AerSimulator(noise_model=nm)
