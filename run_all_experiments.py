"""
run_all_experiments.py
-----------------------
Master entry point: runs all 4 experiments in sequence and produces a
consolidated results summary.

Usage
-----
  python run_all_experiments.py                   # simulate-only (default)
  python run_all_experiments.py --hardware        # use IBM hardware
  python run_all_experiments.py --skip 2 4        # skip experiments 2 and 4
  python run_all_experiments.py --only 1          # run only experiment 1
"""

from __future__ import annotations

import sys
import time
import argparse
import traceback
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run all Hybrid QFT-GD experiments"
    )
    p.add_argument(
        "--hardware",
        action="store_true",
        default=False,
        help="Use real IBM quantum hardware (requires valid ibm_credentials.yaml)",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        type=int,
        default=[],
        metavar="N",
        help="Experiment numbers to skip (e.g. --skip 3 4)",
    )
    p.add_argument(
        "--only",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help="Run only these experiment numbers (e.g. --only 1 2)",
    )
    return p.parse_args()


def run_all(args):
    simulate_only = not args.hardware

    banner = "="*70
    print(f"\n{banner}")
    print("  Hybrid Quantum-Classical Gradient Descent — Full Experiment Suite")
    print(f"  Mode: {'IBM hardware' if not simulate_only else 'Aer simulation'}")
    print(f"{banner}\n")

    # ── Import experiments ──
    from experiments.exp1_convergence       import run_experiment as exp1
    from experiments.exp2_arithmetic_accuracy import run_experiment as exp2
    from experiments.exp3_hardware_validation import run_experiment as exp3
    from experiments.exp4_noise_sensitivity   import run_experiment as exp4

    experiments = {
        1: ("Convergence Analysis",       lambda: exp1(simulate_only=simulate_only)),
        2: ("Arithmetic Accuracy",        lambda: exp2(simulate_only=simulate_only)),
        3: ("Hardware Validation",        lambda: exp3(simulate_only=simulate_only)),
        4: ("Noise Sensitivity Analysis", lambda: exp4()),
    }

    # Filter by --only / --skip
    to_run = sorted(experiments.keys())
    if args.only:
        to_run = [n for n in to_run if n in args.only]
    if args.skip:
        to_run = [n for n in to_run if n not in args.skip]

    print(f"  Running experiments: {to_run}\n")

    results_log = []
    wall_start  = time.time()

    for num in to_run:
        name, fn = experiments[num]
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            status  = "✓ OK"
        except Exception:
            elapsed = time.time() - t0
            status  = "✗ FAILED"
            traceback.print_exc()
        results_log.append((num, name, status, elapsed))

    total_time = time.time() - wall_start

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  Experiment Suite Summary")
    print(f"{'='*70}")
    for num, name, status, elapsed in results_log:
        print(f"  Exp {num}: {name:<35} {status}  ({elapsed:.1f}s)")
    print(f"{'─'*70}")
    print(f"  Total wall time: {total_time:.1f}s")

    # ── Results directory ──
    results_dir = Path("results")
    if results_dir.exists():
        files = sorted(results_dir.glob("*"))
        print(f"\n  Output files in results/:")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<45}  {size_kb:>6.1f} KB")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
