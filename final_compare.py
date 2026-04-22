import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================
# Paths
# ==============================
QUBIT_DIRS = {
    4: "4_qubit_res",
    6: "6_qubit_res",
    8: "8_qubit_res"
}

OUTPUT_DIR = "final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    return pd.read_csv(path)


# ==============================
# 1. Convergence Plot (FIXED)
# ==============================
def plot_convergence():
    plt.figure()

    for q, folder in QUBIT_DIRS.items():
        df = load_csv(os.path.join(folder, "exp1_convergence_summary.csv"))
        if df is None:
            continue

        # Hybrid curve
        plt.plot(df["iteration"], df["hybrid_mean"],
                 label=f"Hybrid ({q} qubits)")

        # Confidence band (VERY NICE)
        plt.fill_between(
            df["iteration"],
            df["hybrid_mean"] - df["hybrid_std"],
            df["hybrid_mean"] + df["hybrid_std"],
            alpha=0.2
        )

        # Classical (only once)
        if q == 4:
            plt.plot(df["iteration"], df["classical_mean"],
                     '--', label="Classical GD")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Convergence: Classical vs Hybrid QFT-GD")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(OUTPUT_DIR, "combined_convergence.png"), dpi=300)
    plt.close()


# ==============================
# 2. Noise Plot (FIXED)
# ==============================
def plot_noise():
    plt.figure()

    for q, folder in QUBIT_DIRS.items():
        df = load_csv(os.path.join(folder, "exp4_noise_sensitivity.csv"))
        if df is None:
            continue

        plt.plot(df["error_rate"], df["mae"],
                 marker='o', label=f"{q} qubits")

    plt.xlabel("Error Rate")
    plt.ylabel("MAE")
    plt.title("Noise Sensitivity vs Qubits")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(OUTPUT_DIR, "combined_noise.png"), dpi=300)
    plt.close()


# ==============================
# 3. Hardware Error Plot (FIXED)
# ==============================
def plot_hardware_error():
    qubits = []
    mae_vals = []

    for q, folder in QUBIT_DIRS.items():
        df = load_csv(os.path.join(folder, "exp3_hardware_validation.csv"))
        if df is None:
            continue

        qubits.append(q)
        mae_vals.append(df["hw_error"].mean())  # ✔ correct column

    plt.figure()
    plt.plot(qubits, mae_vals, marker='o')

    plt.xlabel("Number of Qubits")
    plt.ylabel("Hardware MAE")
    plt.title("Hardware Error vs Qubit Count")
    plt.grid()

    plt.savefig(os.path.join(OUTPUT_DIR, "hardware_vs_qubits.png"), dpi=300)
    plt.close()


# ==============================
# 4. Precision Tradeoff Plot (BEST)
# ==============================
def plot_precision_tradeoff():
    precision_map = {
        4: 0.1333,
        6: 0.0317,
        8: 0.0078
    }

    qubits = []
    mae_vals = []
    precision = []

    for q, folder in QUBIT_DIRS.items():
        df = load_csv(os.path.join(folder, "exp3_hardware_validation.csv"))
        if df is None:
            continue

        qubits.append(q)
        mae_vals.append(df["hw_error"].mean())
        precision.append(precision_map[q])

    plt.figure()
    plt.plot(precision, mae_vals, marker='o')

    plt.xlabel("Precision (LSB)")
    plt.ylabel("Hardware MAE")
    plt.title("Precision vs Hardware Error (Trade-off)")
    plt.gca().invert_xaxis()
    plt.grid()

    plt.savefig(os.path.join(OUTPUT_DIR, "precision_vs_error.png"), dpi=300)
    plt.close()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("Generating all plots...")

    plot_convergence()
    plot_noise()
    plot_hardware_error()
    plot_precision_tradeoff()

    print("✅ All plots saved in 'final_plots/'")