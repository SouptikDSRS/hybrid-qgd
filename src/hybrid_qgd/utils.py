"""
utils.py
--------
Fixed-point encoding/decoding and general helpers for the hybrid QGD framework.

Fixed-point scheme
------------------
A float in [value_min, value_max] is mapped to an integer in [0, 2^n_qubits - 1].
The integer is encoded in the quantum register via X gates (basis encoding).
After measurement the integer is decoded back to float.
"""

from __future__ import annotations

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-point encoding / decoding
# ─────────────────────────────────────────────────────────────────────────────

def float_to_fixed(value: float, n_bits: int, v_min: float = -1.0, v_max: float = 1.0) -> int:
    """Map a float in [v_min, v_max] to an integer in [0, 2^n_bits - 1].

    Parameters
    ----------
    value   : float to encode
    n_bits  : number of qubits (register width)
    v_min   : minimum representable value
    v_max   : maximum representable value

    Returns
    -------
    int in [0, 2^n_bits - 1]
    """
    max_int = (1 << n_bits) - 1
    clipped = float(np.clip(value, v_min, v_max))
    normalized = (clipped - v_min) / (v_max - v_min)   # [0, 1]
    return int(round(normalized * max_int))


def fixed_to_float(integer: int, n_bits: int, v_min: float = -1.0, v_max: float = 1.0) -> float:
    """Decode a fixed-point integer back to a float.

    Parameters
    ----------
    integer : integer in [0, 2^n_bits - 1]
    n_bits  : number of qubits
    v_min   : minimum representable value
    v_max   : maximum representable value

    Returns
    -------
    float in [v_min, v_max]
    """
    max_int = (1 << n_bits) - 1
    integer = int(np.clip(integer, 0, max_int))
    normalized = integer / max_int                       # [0, 1]
    return v_min + normalized * (v_max - v_min)


def encode_value(qc: QuantumCircuit, value: int, reg: QuantumRegister) -> None:
    """Apply X gates to encode an integer into a quantum register (basis encoding).

    The integer is encoded in little-endian order (qubit 0 = LSB).

    Parameters
    ----------
    qc    : QuantumCircuit to append gates to
    value : integer to encode (must fit in len(reg) bits)
    reg   : QuantumRegister to encode into
    """
    n = len(reg)
    for bit in range(n):
        if (value >> bit) & 1:
            qc.x(reg[bit])


def decode_counts(counts: dict[str, int], n_bits: int) -> int:
    """Extract the most-probable measured integer from a counts dict.

    Qiskit returns bit-strings in big-endian order (MSB first), so we
    reverse before converting to int.

    Parameters
    ----------
    counts : {bitstring: count} from a sampler result
    n_bits : expected number of bits

    Returns
    -------
    int — most probable measurement outcome
    """
    best = max(counts, key=counts.get)
    # Strip spaces (sometimes present when multiple registers are measured)
    best = best.replace(" ", "")
    # Qiskit strings are big-endian → reverse for little-endian int conversion
    return int(best[::-1], 2)


def decode_counts_distribution(counts: dict[str, int], n_bits: int,
                                v_min: float = -1.0, v_max: float = 1.0) -> float:
    """Compute the expectation value over the measurement distribution.

    Parameters
    ----------
    counts : {bitstring: count} measurement results
    n_bits : register width
    v_min, v_max : float range for decoding

    Returns
    -------
    float — expectation value
    """
    total_shots = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "")
        integer = int(bitstring[::-1], 2)
        fval = fixed_to_float(integer, n_bits, v_min, v_max)
        exp_val += fval * (count / total_shots)
    return exp_val


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def clip_to_range(value: float, v_min: float = -1.0, v_max: float = 1.0) -> float:
    """Clip a value to [v_min, v_max]."""
    return float(np.clip(value, v_min, v_max))


def precision_of_register(n_bits: int, v_min: float = -1.0, v_max: float = 1.0) -> float:
    """Return the numeric precision (LSB size) of a fixed-point register."""
    return (v_max - v_min) / ((1 << n_bits) - 1)


def mae(a: np.ndarray | list, b: np.ndarray | list) -> float:
    """Mean Absolute Error between two arrays."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def relative_error(predicted: float, reference: float, eps: float = 1e-12) -> float:
    """Relative error |predicted - reference| / (|reference| + eps)."""
    return abs(predicted - reference) / (abs(reference) + eps)
