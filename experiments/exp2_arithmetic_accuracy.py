"""
exp2_arithmetic_accuracy.py
----------------------------
Experiment 2: QFT Arithmetic Accuracy Benchmark
================================================
Randomly samples operand pairs and compares QFT subtraction output
against the classical fixed-point reference.

Metrics
-------
- Mean Absolute Error (MAE)
- Max absolute error
- RMSE
- Accuracy within 1-LSB tolerance
- Error distribution histogram

Output
------
results/exp2_arithmetic_summary.csv
results/exp2_arithmetic_plot.png
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

from src.quantum_backend import BackendManager
from src.qft_arithmetic import QFTArithmetic
from src.utils import mae


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment(simulate_only: bool = True) -> None:
    print("\n" + "="*70)
    print("  EXPERIMENT 2: QFT Arithmetic Accuracy Benchmark")
    print("="*70)

    mgr   = BackendManager("configs/config.yaml", "configs/ibm_credentials.yaml")
    qcfg  = mgr.qft_config
    acfg  = mgr.config.get("arithmetic_experiment", {})

    n_qubits    = qcfg.get("n_qubits", 4)
    shots       = qcfg.get("shots", 4096)
    n_samples   = acfg.get("n_random_samples", 500)
    v_range     = acfg.get("value_range", [-0.9, 0.9])
    v_min_enc   = qcfg.get("value_range", [-1.0, 1.0])[0]
    v_max_enc   = qcfg.get("value_range", [-1.0, 1.0])[1]

    mode    = "simulation" if simulate_only else "hardware"
    backend = mgr.get_backend(mode)
    qft_arith = QFTArithmetic(n_qubits=n_qubits, v_min=v_min_enc, v_max=v_max_enc)

    print(f"  Backend    : {getattr(backend, 'name', str(backend))}")
    print(f"  n_qubits   : {n_qubits}  (precision: {qft_arith.precision:.4f})")
    print(f"  n_samples  : {n_samples}")
    print(f"  shots/circuit: {shots}\n")

    rng = np.random.default_rng(42)
    a_vals = rng.uniform(v_range[0], v_range[1], size=n_samples)
    b_vals = rng.uniform(v_range[0], v_range[1], size=n_samples)

    qft_results      = []
    classical_results = []
    true_results      = []
    errors            = []

    for i in tqdm(range(n_samples), desc="QFT subtract samples"):
        a, b = float(a_vals[i]), float(b_vals[i])
        true_val      = a - b
        classical_val = qft_arith.classical_subtract(a, b)

        # QFT circuit
        qft_val = qft_arith.subtract(a, b, backend, shots=shots)

        qft_results.append(qft_val)
        classical_results.append(classical_val)
        true_results.append(true_val)
        errors.append(abs(qft_val - classical_val))

    qft_results       = np.array(qft_results)
    classical_results = np.array(classical_results)
    true_results      = np.array(true_results)
    errors            = np.array(errors)

    # ── Metrics ──
    lsb = qft_arith.precision
    mae_val  = float(np.mean(errors))
    max_err  = float(np.max(errors))
    rmse     = float(np.sqrt(np.mean(errors**2)))
    within_1lsb = float(np.mean(errors <= lsb) * 100)

    print(f"\n  ─── Results ────────────────────────────────────────────")
    print(f"  LSB (register precision) : {lsb:.4f}")
    print(f"  MAE (QFT vs classical)   : {mae_val:.4f}")
    print(f"  Max absolute error       : {max_err:.4f}")
    print(f"  RMSE                     : {rmse:.4f}")
    print(f"  Within 1-LSB tolerance   : {within_1lsb:.1f}%")

    # ── Save CSV ──
    df = pd.DataFrame({
        "a":              a_vals,
        "b":              b_vals,
        "true_result":    true_results,
        "classical_fp":   classical_results,
        "qft_result":     qft_results,
        "abs_error":      errors,
    })
    csv_path = RESULTS_DIR / "exp2_arithmetic_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV → {csv_path}")

    # ── Metrics row ──
    metrics_df = pd.DataFrame([{
        "n_qubits":    n_qubits,
        "lsb":         lsb,
        "mae":         mae_val,
        "max_error":   max_err,
        "rmse":        rmse,
        "within_1lsb_pct": within_1lsb,
        "n_samples":   n_samples,
        "shots":       shots,
    }])
    metrics_df.to_csv(RESULTS_DIR / "exp2_metrics.csv", index=False)

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Experiment 2: QFT Arithmetic Accuracy  (n_qubits={n_qubits}, shots={shots})",
        fontsize=14, fontweight="bold"
    )

    # Scatter: QFT vs classical
    ax = axes[0]
    ax.scatter(classical_results, qft_results, alpha=0.4, s=12, c=errors,
               cmap="plasma", edgecolors="none")
    lim_min = min(classical_results.min(), qft_results.min()) - 0.05
    lim_max = max(classical_results.max(), qft_results.max()) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.5, label="ideal")
    ax.set_xlabel("Classical fixed-point result", fontsize=11)
    ax.set_ylabel("QFT circuit result",          fontsize=11)
    ax.set_title("QFT vs Classical", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[1]
    ax.hist(errors, bins=40, color="#E74C3C", edgecolor="white", alpha=0.85)
    ax.axvline(mae_val, color="navy", lw=2, linestyle="--", label=f"MAE={mae_val:.4f}")
    ax.axvline(lsb, color="green", lw=1.5, linestyle=":", label=f"1-LSB={lsb:.4f}")
    ax.set_xlabel("Absolute Error", fontsize=11)
    ax.set_ylabel("Count",          fontsize=11)
    ax.set_title("Error Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Error vs magnitude of operands
    ax = axes[2]
    magnitudes = np.abs(a_vals - b_vals)
    ax.scatter(magnitudes, errors, alpha=0.4, s=12, color="#1ABC9C", edgecolors="none")
    ax.set_xlabel("|a - b| (true difference)", fontsize=11)
    ax.set_ylabel("Absolute Error",            fontsize=11)
    ax.set_title("Error vs Operand Magnitude", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp2_arithmetic_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    run_experiment(simulate_only=not args.hardware)
