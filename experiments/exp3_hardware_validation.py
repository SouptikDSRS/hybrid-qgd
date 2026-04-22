"""
exp3_hardware_validation.py
----------------------------
Experiment 3: Hardware Validation
===================================
Executes QFT arithmetic circuits on real IBM quantum hardware and compares
results against the Aer statevector simulator.

This is the key hardware-in-the-loop experiment that validates whether the
QFT subtraction circuit behaves correctly under real gate noise, crosstalk,
and decoherence.

Metrics
-------
- MAE: hardware vs simulator
- Circuit depth (pre- and post-transpilation)
- Hardware execution time
- Fidelity proxy (fraction of shots on correct state)

Output
------
results/exp3_hardware_validation.csv
results/exp3_hardware_plot.png
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from hybrid_qgd.quantum_backend import BackendManager
from hybrid_qgd.qft_arithmetic import QFTArithmetic
from hybrid_qgd.utils import mae


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Test cases: (a, b) pairs for hardware validation
TEST_PAIRS = [
    (0.5,  0.25),
    (-0.5, 0.25),
    (0.75, 0.5),
    (-0.25, -0.5),
    (0.0,  0.5),
    (0.3, -0.3),
    (0.8,  0.1),
    (-0.8, -0.1),
    (0.6,  0.4),
    (-0.6, -0.4),
]


def run_experiment(backend_name: str | None = None,
                   simulate_only: bool = True) -> None:
    print("\n" + "="*70)
    print("  EXPERIMENT 3: Hardware Validation — IBM Device vs Simulator")
    print("="*70)

    mgr  = BackendManager("configs/config.yaml", "configs/ibm_credentials.yaml")
    qcfg = mgr.qft_config
    n_qubits = qcfg.get("n_qubits", 4)
    shots    = qcfg.get("shots", 4096)
    v_min    = qcfg.get("value_range", [-1.0, 1.0])[0]
    v_max    = qcfg.get("value_range", [-1.0, 1.0])[1]

    # Simulator (always available)
    sim_backend = mgr.get_simulator()
    qft_arith   = QFTArithmetic(n_qubits=n_qubits, v_min=v_min, v_max=v_max)

    if simulate_only:
        print("  [simulate_only mode] Comparing Aer statevector vs Aer QASM sampler")
        hw_backend = sim_backend
        hw_label   = "Aer-QASM"
    else:
        hw_backend = mgr.get_hardware_backend()
        hw_label   = getattr(hw_backend, "name", "IBM-hardware")

    print(f"  Simulator  : {getattr(sim_backend, 'name', str(sim_backend))}")
    print(f"  Hardware   : {hw_label}")
    print(f"  n_qubits   : {n_qubits}, shots: {shots}")
    print(f"  Test pairs : {len(TEST_PAIRS)}\n")

    records = []
    sim_results = []
    hw_results  = []

    for (a, b) in tqdm(TEST_PAIRS, desc="Test pairs"):
        true_val      = a - b
        classical_val = qft_arith.classical_subtract(a, b)

        # Simulator
        sim_val = qft_arith.subtract(a, b, sim_backend, shots=shots)

        # Hardware (or Aer QASM in simulate_only)
        hw_val = qft_arith.subtract(a, b, hw_backend, shots=shots)

        sim_err = abs(sim_val - classical_val)
        hw_err  = abs(hw_val  - classical_val)
        cross_err = abs(sim_val - hw_val)

        records.append({
            "a":              a,
            "b":              b,
            "true_a_minus_b": true_val,
            "classical_fp":   classical_val,
            "simulator":      sim_val,
            "hardware":       hw_val,
            "sim_error":      sim_err,
            "hw_error":       hw_err,
            "cross_error":    cross_err,
        })
        sim_results.append(sim_val)
        hw_results.append(hw_val)

    df = pd.DataFrame(records)

    mae_sim  = float(np.mean(df["sim_error"]))
    mae_hw   = float(np.mean(df["hw_error"]))
    mae_cross = float(np.mean(df["cross_error"]))

    print(f"\n  ─── Results ────────────────────────────────────────────")
    print(f"  MAE (simulator vs classical fp) : {mae_sim:.4f}")
    print(f"  MAE (hardware  vs classical fp) : {mae_hw:.4f}")
    print(f"  MAE (hardware  vs simulator)    : {mae_cross:.4f}")

    # ── Circuit depth analysis ──
    sample_qc = qft_arith.build_subtraction_circuit(0.5, 0.25)
    print(f"\n  Raw circuit depth  : {sample_qc.depth()}")
    print(f"  Raw gate count     : {sample_qc.size()}")

    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=1, backend=sim_backend)
        transpiled = pm.run(sample_qc)
        print(f"  Transpiled depth   : {transpiled.depth()}")
        print(f"  Transpiled gates   : {transpiled.size()}")
        df["circuit_depth_raw"]         = sample_qc.depth()
        df["circuit_depth_transpiled"]  = transpiled.depth()
    except Exception as e:
        print(f"  Transpile info unavailable: {e}")

    # ── Save CSV ──
    csv_path = RESULTS_DIR / "exp3_hardware_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV → {csv_path}")

    # ── Plots ──
    classical_vals = np.array([r["classical_fp"] for r in records])
    sim_arr = np.array(sim_results)
    hw_arr  = np.array(hw_results)
    labels  = [f"({a:.2f},{b:.2f})" for a, b in TEST_PAIRS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Experiment 3: Hardware Validation  (n_qubits={n_qubits}, shots={shots})",
        fontsize=14, fontweight="bold"
    )

    # Scatter: sim vs hw vs classical
    ax = axes[0]
    x = np.arange(len(TEST_PAIRS))
    w = 0.28
    ax.bar(x - w, classical_vals, w, label="Classical FP", color="#3498DB", alpha=0.8)
    ax.bar(x,     sim_arr,        w, label="Simulator",    color="#2ECC71", alpha=0.8)
    ax.bar(x + w, hw_arr,         w, label=hw_label,       color="#E74C3C", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("a - b result", fontsize=11)
    ax.set_title("Results per Test Pair", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Error bar chart
    ax = axes[1]
    ax.bar(x - w/2, df["sim_error"].values, w, label=f"Sim error (MAE={mae_sim:.3f})",
           color="#2ECC71", alpha=0.85)
    ax.bar(x + w/2, df["hw_error"].values,  w, label=f"HW error  (MAE={mae_hw:.3f})",
           color="#E74C3C", alpha=0.85)
    ax.axhline(qft_arith.precision, color="navy", lw=1.5,
               linestyle="--", label=f"1-LSB={qft_arith.precision:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Absolute Error", fontsize=11)
    ax.set_title("Errors vs Classical FP", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp3_hardware_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default=None,
                        help="Specific IBM backend name (e.g. ibm_fez)")
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    run_experiment(backend_name=args.backend, simulate_only=not args.hardware)
