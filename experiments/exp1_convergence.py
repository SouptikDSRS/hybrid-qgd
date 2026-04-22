"""
exp1_convergence.py
--------------------
Experiment 1: Convergence Analysis
===================================
Compares Classical Gradient Descent vs Hybrid QFT-GD over N≥10 independent
trials with randomised parameter initialisation.

Metrics
-------
- Mean loss per iteration
- Standard deviation of loss across trials
- Mean convergence iteration
- Final loss distribution

Output
------
results/exp1_convergence_summary.csv
results/exp1_convergence_plot.png
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

from src.quantum_backend import BackendManager
from src.hybrid_optimizer import HybridQFTOptimizer, ClassicalGradientDescent, OptimizationResult
from src.objective_functions import get_function, get_gradient


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = "configs/config.yaml"
CREDS_PATH  = "configs/ibm_credentials.yaml"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment(simulate_only: bool = True) -> None:
    print("\n" + "="*70)
    print("  EXPERIMENT 1: Convergence Analysis — Classical GD vs Hybrid QFT-GD")
    print("="*70)

    # ── Backend ──
    mgr = BackendManager(CONFIG_PATH, CREDS_PATH)
    cfg = mgr.optimizer_config
    qcfg = mgr.qft_config

    mode = "simulation" if simulate_only else "hardware"
    backend = mgr.get_backend(mode)
    print(f"  Backend: {getattr(backend, 'name', str(backend))}")

    # ── Hyperparameters ──
    lr           = cfg.get("learning_rate", 0.1)
    max_iter     = cfg.get("max_iterations", 50)
    n_trials     = cfg.get("n_trials", 10)
    conv_tol     = float(cfg.get("convergence_threshold", 1e-6))
    n_qubits     = qcfg.get("n_qubits", 4)
    shots        = qcfg.get("shots", 4096)
    seed         = cfg.get("random_seed", 42)
    obj_name     = mgr.config.get("objective", {}).get("function", "quadratic")
    n_params     = mgr.config.get("objective", {}).get("n_params", 2)

    rng = np.random.default_rng(seed)
    loss_fn = get_function(obj_name)
    grad_fn = get_gradient(obj_name)

    print(f"\n  Config: lr={lr}, max_iter={max_iter}, n_trials={n_trials}")
    print(f"  Objective: {obj_name} ({n_params} params)")
    print(f"  QFT: n_qubits={n_qubits}, shots={shots}\n")

    # ── Collect results ──
    classical_losses: list[list[float]] = []
    hybrid_losses:    list[list[float]] = []

    for trial in tqdm(range(n_trials), desc="Trials"):
        init = rng.uniform(-0.8, 0.8, size=n_params)

        # Classical GD
        cgd = ClassicalGradientDescent(
            loss_fn=loss_fn,
            gradient_fn=grad_fn,
            learning_rate=lr,
            max_iterations=max_iter,
            convergence_tol=conv_tol,
            verbose=False,
        )
        c_result = cgd.optimize(init.copy())
        classical_losses.append(c_result.loss_history)

        # Hybrid QFT-GD
        hgd = HybridQFTOptimizer(
            loss_fn=loss_fn,
            backend=backend,
            learning_rate=lr,
            n_qubits=n_qubits,
            v_min=-2.0,
            v_max=2.0,
            shots=shots,
            max_iterations=max_iter,
            convergence_tol=conv_tol,
            mode="hybrid",
            verbose=False,
        )
        h_result = hgd.optimize(init.copy())
        hybrid_losses.append(h_result.loss_history)

    # ── Pad to equal length ──
    max_len = max(
        max(len(l) for l in classical_losses),
        max(len(l) for l in hybrid_losses),
    )

    def pad(history_list: list[list[float]], length: int) -> np.ndarray:
        padded = []
        for h in history_list:
            arr = np.array(h, dtype=float)
            if len(arr) < length:
                arr = np.concatenate([arr, np.full(length - len(arr), arr[-1])])
            padded.append(arr[:length])
        return np.array(padded)

    c_arr = pad(classical_losses, max_len)
    h_arr = pad(hybrid_losses,    max_len)

    c_mean, c_std = c_arr.mean(axis=0), c_arr.std(axis=0)
    h_mean, h_std = h_arr.mean(axis=0), h_arr.std(axis=0)
    iters = np.arange(max_len)

    # ── Save CSV ──
    df = pd.DataFrame({
        "iteration":       iters,
        "classical_mean":  c_mean,
        "classical_std":   c_std,
        "hybrid_mean":     h_mean,
        "hybrid_std":      h_std,
    })
    csv_path = RESULTS_DIR / "exp1_convergence_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved CSV → {csv_path}")

    # ── Plot ──
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left: convergence curves
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(iters, c_mean, "b-",  lw=2, label="Classical GD")
    ax1.fill_between(iters, c_mean - c_std, c_mean + c_std, alpha=0.2, color="blue")
    ax1.plot(iters, h_mean, "r--", lw=2, label="Hybrid QFT-GD")
    ax1.fill_between(iters, h_mean - h_std, h_mean + h_std, alpha=0.2, color="red")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title(f"Convergence: {obj_name}", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Right: final loss distributions
    ax2 = fig.add_subplot(gs[1])
    c_final = c_arr[:, -1]
    h_final = h_arr[:, -1]
    ax2.boxplot([c_final, h_final],
                labels=["Classical GD", "Hybrid QFT-GD"],
                patch_artist=True,
                boxprops=dict(facecolor="#AED6F1"),
                medianprops=dict(color="navy", lw=2))
    ax2.set_ylabel("Final Loss", fontsize=12)
    ax2.set_title("Final Loss Distribution", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Experiment 1: Classical GD vs Hybrid QFT-GD  [{n_trials} trials]",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plot_path = RESULTS_DIR / "exp1_convergence_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {plot_path}")

    # ── Summary ──
    print(f"\n  ─── Summary ───────────────────────────────────────────")
    print(f"  Classical GD  final loss: {c_mean[-1]:.4f} ± {c_std[-1]:.4f}")
    print(f"  Hybrid QFT-GD final loss: {h_mean[-1]:.4f} ± {h_std[-1]:.4f}")
    print(f"  Convergence iterations (classical): "
          f"{np.mean([len(l) for l in classical_losses]):.1f}")
    print(f"  Convergence iterations (hybrid):    "
          f"{np.mean([len(l) for l in hybrid_losses]):.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-only", action="store_true", default=True)
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    simulate_only = not args.hardware
    run_experiment(simulate_only=simulate_only)
