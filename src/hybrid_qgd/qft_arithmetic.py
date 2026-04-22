"""
qft_arithmetic.py
-----------------
QFT-based quantum arithmetic circuits using the Draper addition algorithm.

Theory
------
The Draper algorithm (quant-ph/0008033) performs addition in the Fourier basis:
  1. Encode integer a into register (big-endian: qubit 0 = MSB)
  2. Apply QFT (no output-swap variant, qubit 0 remains MSB after transform)
  3. Add integer b via single-qubit phase rotations:
       qubit j → angle += 2π · b / 2^(n-j)
  4. Apply inverse QFT
  5. Measure; decode result

Subtraction: negate all phase rotation angles.

Encoding
--------
Floats in [v_min, v_max] → integers in [0, 2^n - 1] via uniform fixed-point
quantisation.  Rounding error ≤ 1 LSB = (v_max - v_min) / (2^n - 1).

Execution
---------
All circuits use native gates (H, CP, P, X) — no high-level library gates —
to ensure Aer and IBM Runtime compatibility.
"""

from __future__ import annotations

import math
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from hybrid_qgd.utils import float_to_fixed, fixed_to_float


# ─────────────────────────────────────────────────────────────────────────────
# Native gate QFT / IQFT (big-endian, no output swap)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_qft(qc, qr):
    """QFT (no swap), qubit 0 = MSB convention."""
    n = len(qr)
    for i in range(n):
        qc.h(qr[i])
        for j in range(i + 1, n):
            qc.cp(2.0 * math.pi / (1 << (j - i + 1)), qr[i], qr[j])


def _apply_iqft(qc, qr):
    """Inverse QFT (no swap), qubit 0 = MSB convention."""
    n = len(qr)
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            qc.cp(-2.0 * math.pi / (1 << (j - i + 1)), qr[i], qr[j])
        qc.h(qr[i])


def _apply_phase_add(qc, qr, k, negate=False):
    """Draper phase rotations: add (or subtract) integer k in the QFT basis.

    For no-swap QFT with qubit 0 = MSB:
      angle for qubit j = ±2π · k / 2^(n - j)
    """
    n   = len(qr)
    sgn = -1 if negate else 1
    for j in range(n):
        angle = sgn * 2.0 * math.pi * k / (1 << (n - j))
        qc.p(angle, qr[j])


def _encode_big_endian(qc, integer, qr):
    """X-gate encoding of integer into qr, qubit 0 = MSB."""
    n = len(qr)
    for bit in range(n):
        if (integer >> (n - 1 - bit)) & 1:
            qc.x(qr[bit])


def _decode_distribution_be(counts, n, v_min, v_max):
    """Expectation value over measurement distribution (big-endian encoding).

    Qiskit measurement strings are q[n-1]...q[0] (qubit 0 = rightmost char).
    With big-endian encoding (qubit 0 = MSB): reversing the string gives
    the integer in standard binary (MSB first).
    """
    total   = sum(counts.values())
    exp_val = 0.0
    for bitstring, cnt in counts.items():
        bitstring = bitstring.replace(" ", "")
        integer   = int(bitstring[::-1], 2)       # little-endian interpretation
        fval      = fixed_to_float(integer, n, v_min, v_max)
        exp_val  += fval * (cnt / total)
    return exp_val


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class QFTArithmetic:
    """QFT-based fixed-point arithmetic for the hybrid gradient descent loop.

    Parameters
    ----------
    n_qubits : register width
    v_min    : minimum representable float
    v_max    : maximum representable float
    """

    def __init__(self, n_qubits=4, v_min=-1.0, v_max=1.0):
        self.n     = n_qubits
        self.v_min = v_min
        self.v_max = v_max

    def _encode(self, value):
        return float_to_fixed(value, self.n, self.v_min, self.v_max)

    def _decode(self, integer):
        return fixed_to_float(integer, self.n, self.v_min, self.v_max)

    def build_subtraction_circuit(self, a, b):
        """Build a QFT circuit computing a - b."""
        int_a = self._encode(a)
        int_b = self._encode(b)
        qr = QuantumRegister(self.n, "a")
        cr = ClassicalRegister(self.n, "c")
        qc = QuantumCircuit(qr, cr, name="QFT_SUB")
        _encode_big_endian(qc, int_a, qr)
        _apply_qft(qc, qr)
        _apply_phase_add(qc, qr, int_b, negate=True)
        _apply_iqft(qc, qr)
        qc.measure(qr, cr)
        return qc

    def build_addition_circuit(self, a, b):
        """Build a QFT circuit computing a + b."""
        int_a = self._encode(a)
        int_b = self._encode(b)
        qr = QuantumRegister(self.n, "a")
        cr = ClassicalRegister(self.n, "c")
        qc = QuantumCircuit(qr, cr, name="QFT_ADD")
        _encode_big_endian(qc, int_a, qr)
        _apply_qft(qc, qr)
        _apply_phase_add(qc, qr, int_b, negate=False)
        _apply_iqft(qc, qr)
        qc.measure(qr, cr)
        return qc

    def classical_subtract(self, a, b):
        """Fixed-point roundtrip of (a - b) — matches quantum rounding."""
        int_a      = self._encode(a)
        int_b      = self._encode(b)
        result_int = (int_a - int_b) % (1 << self.n)
        return self._decode(result_int)

    def classical_add(self, a, b):
        """Fixed-point roundtrip of (a + b)."""
        int_a      = self._encode(a)
        int_b      = self._encode(b)
        result_int = (int_a + int_b) % (1 << self.n)
        return self._decode(result_int)

    def execute_circuit(self, qc, backend, shots=4096):
        """Run a QFT circuit on backend and return decoded float."""
        from qiskit_aer import AerSimulator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm         = generate_preset_pass_manager(optimization_level=1, backend=backend)
        transpiled = pm.run(qc)

        if isinstance(backend, AerSimulator):
            from qiskit_aer.primitives import SamplerV2 as AerSampler
            sampler = AerSampler.from_backend(backend)
        else:
            from qiskit_ibm_runtime import SamplerV2
            sampler = SamplerV2(backend)

        job    = sampler.run([transpiled], shots=shots)
        result = job.result()
        counts = result[0].data.c.get_counts()
        return _decode_distribution_be(counts, self.n, self.v_min, self.v_max)

    def subtract(self, a, b, backend, shots=4096):
        """End-to-end QFT subtraction: build → run → decode."""
        return self.execute_circuit(self.build_subtraction_circuit(a, b), backend, shots)

    def add(self, a, b, backend, shots=4096):
        """End-to-end QFT addition: build → run → decode."""
        return self.execute_circuit(self.build_addition_circuit(a, b), backend, shots)

    @property
    def precision(self):
        """LSB size: smallest representable difference."""
        from hybrid_qgd.utils import precision_of_register
        return precision_of_register(self.n, self.v_min, self.v_max)

    def circuit_info(self):
        qc = self.build_subtraction_circuit(0.5, 0.25)
        return {"n_qubits": self.n, "depth": qc.depth(), "gate_count": qc.size()}

    def __repr__(self):
        return (f"QFTArithmetic(n_qubits={self.n}, "
                f"range=[{self.v_min}, {self.v_max}], "
                f"precision={self.precision:.4f})")
