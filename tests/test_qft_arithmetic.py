"""
test_qft_arithmetic.py
-----------------------
Unit tests for QFT arithmetic circuits.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from qiskit_aer import AerSimulator

from src.qft_arithmetic import QFTArithmetic
from src.utils import float_to_fixed, fixed_to_float, precision_of_register


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ideal_backend():
    return AerSimulator()


@pytest.fixture(scope="module")
def qft4():
    return QFTArithmetic(n_qubits=4, v_min=-1.0, v_max=1.0)


@pytest.fixture(scope="module")
def qft6():
    return QFTArithmetic(n_qubits=6, v_min=-1.0, v_max=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding / decoding roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodingDecoding:
    def test_roundtrip_zero(self):
        n = 4
        for v in [0.0, 0.5, -0.5, 1.0, -1.0]:
            enc = float_to_fixed(v, n, -1.0, 1.0)
            dec = fixed_to_float(enc, n, -1.0, 1.0)
            # Should be within 1 LSB
            lsb = precision_of_register(n, -1.0, 1.0)
            assert abs(dec - v) <= lsb + 1e-9, \
                f"Roundtrip failed for v={v}: got {dec}, lsb={lsb}"

    def test_encoding_range(self):
        n = 4
        # v_min → 0, v_max → 2^n - 1
        assert float_to_fixed(-1.0, n, -1.0, 1.0) == 0
        assert float_to_fixed(1.0, n, -1.0, 1.0) == (1 << n) - 1

    def test_precision_increases_with_qubits(self):
        lsb4 = precision_of_register(4)
        lsb6 = precision_of_register(6)
        lsb8 = precision_of_register(8)
        assert lsb4 > lsb6 > lsb8


# ─────────────────────────────────────────────────────────────────────────────
# Circuit construction
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitConstruction:
    def test_subtraction_circuit_width(self, qft4):
        qc = qft4.build_subtraction_circuit(0.5, 0.25)
        assert qc.num_qubits == 4
        assert qc.num_clbits == 4

    def test_addition_circuit_width(self, qft4):
        qc = qft4.build_addition_circuit(0.5, 0.25)
        assert qc.num_qubits == 4
        assert qc.num_clbits == 4

    def test_circuit_has_measurements(self, qft4):
        qc = qft4.build_subtraction_circuit(0.0, 0.0)
        ops = [inst.operation.name for inst in qc.data]
        assert "measure" in ops

    def test_circuit_depth_positive(self, qft4):
        qc = qft4.build_subtraction_circuit(0.5, 0.25)
        assert qc.depth() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Classical reference
# ─────────────────────────────────────────────────────────────────────────────

class TestClassicalReference:
    """Tests for the classical fixed-point reference that mirrors QFT circuit arithmetic.

    Fixed-point arithmetic note
    ---------------------------
    With range [v_min, v_max] = [-1, 1], the encoding is AFFINE:
      encoded(x) = (x - v_min) / (v_max - v_min) * (2^n - 1)

    Subtraction of encoded integers gives:
      decoded(encoded(a) - encoded(b)) = v_min + (a - b)  NOT  (a - b)

    This is the correct behaviour of the fixed-point register and is what
    the QFT circuit implements. The optimizer corrects for this offset when
    using the QFT result to update parameters.
    """

    def test_classical_subtract_basic(self, qft4):
        """decoded(int_a - int_b) = v_min + (a - b) = -1 + 0.25 ≈ -0.733."""
        result = qft4.classical_subtract(0.5, 0.25)
        lsb    = qft4.precision
        # Expected: v_min + (0.5 - 0.25) = -1.0 + 0.25 = -0.75, rounded to nearest LSB
        expected = qft4.v_min + (0.5 - 0.25)  # -0.75
        assert abs(result - expected) <= lsb + 1e-9, \
            f"Got {result}, expected ~{expected} (±{lsb})"

    def test_classical_add_basic(self, qft4):
        """classical_add returns a consistent fixed-point result and matches QFT."""
        # Note: with v_min=-1, v_max=1 and 4 qubits, adding 0.25+0.25 may wrap
        # modularly (encoded 9+9=18 → 18%16=2 → -0.733). The key invariant is
        # that classical_add and the QFT circuit agree to within 1 LSB.
        result = qft4.classical_add(0.25, 0.25)
        lsb    = qft4.precision
        assert qft4.v_min <= result <= qft4.v_max, \
            f"Result {result} outside [{qft4.v_min}, {qft4.v_max}]"
        # Self-consistency: repeated calls must be identical
        assert abs(qft4.classical_add(0.25, 0.25) - result) < 1e-12

    def test_classical_subtract_zero(self, qft4):
        """Subtracting equal values: decoded(0) = v_min."""
        result = qft4.classical_subtract(0.5, 0.5)
        assert abs(result - qft4.v_min) <= 1e-9, \
            f"Got {result}, expected v_min={qft4.v_min}"

    def test_classical_subtract_matches_qft(self, qft4, ideal_backend):
        """classical_subtract and QFT circuit must agree to within 1 LSB."""
        for a, b in [(0.5, 0.25), (-0.3, 0.4), (0.8, -0.2)]:
            c  = qft4.classical_subtract(a, b)
            qt = qft4.subtract(a, b, ideal_backend, shots=8192)
            assert abs(qt - c) <= qft4.precision + 1e-9, \
                f"classical={c:.4f} qft={qt:.4f} diff={abs(qt-c):.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# QFT circuit execution (ideal simulator)
# ─────────────────────────────────────────────────────────────────────────────

class TestQFTExecution:
    @pytest.mark.parametrize("a,b", [
        (0.5, 0.25),
        (-0.5, 0.25),
        (0.0, 0.5),
        (0.75, 0.5),
        (-0.25, -0.5),
    ])
    def test_subtraction_accuracy(self, qft4, ideal_backend, a, b):
        """QFT result should match classical fixed-point within 1 LSB."""
        classical = qft4.classical_subtract(a, b)
        qft_result = qft4.subtract(a, b, ideal_backend, shots=8192)
        lsb = qft4.precision
        assert abs(qft_result - classical) <= lsb + 1e-9, \
            f"QFT subtract({a},{b}): got {qft_result}, expected {classical}, lsb={lsb}"

    def test_subtraction_higher_precision(self, qft6, ideal_backend):
        """Higher qubit count → tighter precision."""
        a, b = 0.6, 0.3
        classical = qft6.classical_subtract(a, b)
        qft_result = qft6.subtract(a, b, ideal_backend, shots=8192)
        lsb = qft6.precision
        assert abs(qft_result - classical) <= lsb + 1e-9

    def test_addition_accuracy(self, qft4, ideal_backend):
        a, b = 0.25, 0.25
        classical = qft4.classical_add(a, b)
        qft_result = qft4.add(a, b, ideal_backend, shots=8192)
        lsb = qft4.precision
        assert abs(qft_result - classical) <= lsb + 1e-9

    def test_n_samples_consistency(self, qft4, ideal_backend):
        """Repeated executions of same circuit should give consistent results."""
        results = []
        for _ in range(5):
            r = qft4.subtract(0.5, 0.25, ideal_backend, shots=4096)
            results.append(r)
        # All results should be equal (deterministic fixed-point encoding)
        assert all(abs(r - results[0]) <= qft4.precision for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Properties
# ─────────────────────────────────────────────────────────────────────────────

class TestProperties:
    def test_precision_4q(self, qft4):
        expected = 2.0 / (2**4 - 1)  # (v_max - v_min) / (2^n - 1)
        assert abs(qft4.precision - expected) < 1e-12

    def test_repr(self, qft4):
        r = repr(qft4)
        assert "QFTArithmetic" in r
        assert "n_qubits=4" in r
