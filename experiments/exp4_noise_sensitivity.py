"""
exp4_noise_sensitivity.py
--------------------------
Experiment 4: Noise Sensitivity Analysis
=========================================
Sweeps depolarizing error rates and measures how QFT arithmetic accuracy
degrades as noise increases. Models the realistic NISQ noise environment.

Metrics per noise level
-----------------------
- MAE vs classical fixed-point reference
- RMSE
- Fraction within 1-LSB tolerance
- Mean circuit fidelity proxy

Output
------
results/exp4_noise_sensitivity.csv
results/exp4_noise_plot.png
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

from qiskit_aer import AerSimulator

from hybrid_qgd.quantum_backend import BackendManager
from hybrid_qgd.qft_arithmetic import QFTArithmetic
from hybrid_qgd.noise_model import build_combined_noise_model


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment() -> None:
    print("\n" + "="*70)
    print("  EXPERIMENT 4: Noise Sensitivity Analysis")
    print("="*70)

    mgr  = BackendManager("configs/config.yaml", "configs/ibm_credentials.yaml")
    qcfg = mgr.qft_config
    ncfg = mgr.config.get("noise_experiment", {})

    n_qubits    = qcfg.get("n_qubits", 4)
    shots       = qcfg.get("shots", 4096)
    v_min       = qcfg.get("value_range", [-1.0, 1.0])[0]
    v_max       = qcfg.get("value_range", [-1.0, 1.0])[1]
    error_rates = ncfg.get("error_rates", [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    n_per_rate  = ncfg.get("n_samples_per_rate", 200)

    qft_arith = QFTArithmetic(n_qubits=n_qubits, v_min=v_min, v_max=v_max)
    lsb = qft_arith.precision

    print(f"  n_qubits    : {n_qubits}  (LSB={lsb:.4f})")
    print(f"  shots       : {shots}")
    print(f"  error_rates : {error_rates}")
    print(f"  n_per_rate  : {n_per_rate}\n")

    rng = np.random.default_rng(42)
    a_vals = rng.uniform(-0.85, 0.85, n_per_rate)
    b_vals = rng.uniform(-0.85, 0.85, n_per_rate)

    records = []

    for rate in tqdm(error_rates, desc="Error rates"):
        # Build noisy backend
        if rate == 0.0:
            backend = AerSimulator()   # ideal
        else:
            nm = build_combined_noise_model(
                depolarizing_rate=rate,
                readout_error=rate * 5,   # readout ~5× gate error
            )
            backend = AerSimulator(noise_model=nm)

        errors = []
        for a, b in zip(a_vals, b_vals):
            classical_val = qft_arith.classical_subtract(float(a), float(b))
            qft_val       = qft_arith.subtract(float(a), float(b), backend, shots=shots)
            errors.append(abs(qft_val - classical_val))

        errors = np.array(errors)
        mae_val       = float(np.mean(errors))
        rmse          = float(np.sqrt(np.mean(errors**2)))
        within_1lsb   = float(np.mean(errors <= lsb) * 100)
        max_err       = float(np.max(errors))

        records.append({
            "error_rate":       rate,
            "mae":              mae_val,
            "rmse":             rmse,
            "max_error":        max_err,
            "within_1lsb_pct":  within_1lsb,
            "n_samples":        n_per_rate,
        })

        print(f"  rate={rate:.3f}  MAE={mae_val:.4f}  "
              f"within-1LSB={within_1lsb:.1f}%  RMSE={rmse:.4f}")

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "exp4_noise_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV → {csv_path}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Experiment 4: Noise Sensitivity  (n_qubits={n_qubits}, shots={shots})",
        fontsize=14, fontweight="bold"
    )

    rates = df["error_rate"].values

    # MAE vs error rate
    ax = axes[0]
    ax.plot(rates, df["mae"].values,  "o-", color="#E74C3C", lw=2,
            markersize=7, label="MAE")
    ax.plot(rates, df["rmse"].values, "s--", color="#3498DB", lw=2,
            markersize=6, label="RMSE")
    ax.axhline(lsb, color="green", lw=1.5, linestyle=":", label=f"1-LSB={lsb:.3f}")
    ax.set_xlabel("Depolarizing Error Rate", fontsize=11)
    ax.set_ylabel("Arithmetic Error",        fontsize=11)
    ax.set_title("MAE & RMSE vs Noise",     fontsize=12)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Within-1LSB accuracy vs noise
    ax = axes[1]
    ax.plot(rates, df["within_1lsb_pct"].values, "D-", color="#9B59B6",
            lw=2, markersize=7)
    ax.axhline(100.0, color="gray", lw=1, linestyle="--", alpha=0.6)
    ax.fill_between(rates, df["within_1lsb_pct"].values, 0,
                    alpha=0.15, color="#9B59B6")
    ax.set_xlabel("Depolarizing Error Rate",    fontsize=11)
    ax.set_ylabel("% Results within 1-LSB",     fontsize=11)
    ax.set_title("Accuracy vs Noise Level",     fontsize=12)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Max error vs noise
    ax = axes[2]
    ax.plot(rates, df["max_error"].values, "^-", color="#E67E22",
            lw=2, markersize=7, label="Max error")
    ax.axhline(lsb, color="green", lw=1.5, linestyle=":", label=f"1-LSB={lsb:.3f}")
    ax.axhline(2 * lsb, color="orange", lw=1.5, linestyle="--",
               label=f"2-LSB={2*lsb:.3f}")
    ax.set_xlabel("Depolarizing Error Rate",   fontsize=11)
    ax.set_ylabel("Max Absolute Error",        fontsize=11)
    ax.set_title("Worst-case Error vs Noise",  fontsize=12)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp4_noise_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {plot_path}")

    # ── Summary table ──
    print(f"\n  ─── Noise Sensitivity Summary ──────────────────────────")
    print(df.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    run_experiment()
