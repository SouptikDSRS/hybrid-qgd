# Hybrid Quantum-Classical Gradient Descent via QFT-Based Arithmetic

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Qiskit 2.x](https://img.shields.io/badge/qiskit-2.x-purple.svg)](https://qiskit.org/)
[![IBM Quantum](https://img.shields.io/badge/IBM-Quantum-blue.svg)](https://quantum.ibm.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements a **hybrid quantum-classical optimization framework** that integrates QFT-based quantum arithmetic directly into the gradient descent update loop. It is designed for real execution on IBM Quantum hardware (IBM Fez, IBM Torino) as well as GPU-accelerated simulation via Qiskit Aer.

### Key Innovation

Classical gradient descent updates parameters as:

```
x_{t+1} = x_t - α · g_t
```

This project replaces the **subtraction step** with a **QFT-based quantum arithmetic circuit**, while gradients are computed via the **Parameter Shift Rule (PSR)**. The framework is fully modular, hardware-aware, and noise-characterized.

---

## Project Structure

```
hybrid_qgd/
├── src/
│   ├── __init__.py
│   ├── qft_arithmetic.py        # QFT-based addition/subtraction circuits
│   ├── parameter_shift.py       # Gradient computation via PSR
│   ├── hybrid_optimizer.py      # Core hybrid GD optimizer
│   ├── quantum_backend.py       # IBM Runtime + Aer backend manager
│   ├── noise_model.py           # Noise model construction & characterization
│   ├── objective_functions.py   # Test objective functions
│   └── utils.py                 # Encoding, decoding, helpers
├── experiments/
│   ├── exp1_convergence.py      # Convergence: Classical vs Hybrid QFT-GD
│   ├── exp2_arithmetic_accuracy.py  # QFT arithmetic accuracy benchmark
│   ├── exp3_hardware_validation.py  # Real hardware vs simulator
│   └── exp4_noise_sensitivity.py    # Noise level sweep analysis
├── configs/
│   ├── config.yaml              # Global configuration
│   └── ibm_credentials.yaml     # IBM Quantum credentials (gitignored)
├── tests/
│   ├── test_qft_arithmetic.py
│   ├── test_parameter_shift.py
│   └── test_hybrid_optimizer.py
├── results/                     # Auto-generated experiment outputs
├── notebooks/
│   └── demo.ipynb               # Interactive walkthrough
├── docs/
│   └── architecture.md
├── run_all_experiments.py       # Single entry point for all experiments
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourname/hybrid_qgd.git
cd hybrid_qgd

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure IBM Quantum credentials
cp configs/ibm_credentials.yaml.example configs/ibm_credentials.yaml
# Edit ibm_credentials.yaml and add your IBM Quantum API token
```

---

## IBM Quantum Setup

1. Create an account at [quantum.ibm.com](https://quantum.ibm.com/)
2. Copy your API token
3. Add it to `configs/ibm_credentials.yaml`:

```yaml
ibm_token: "YOUR_TOKEN_HERE"
channel: "ibm_quantum"
instance: "ibm-q/open/main"
preferred_backends:
  - ibm_fez
  - ibm_torino
  - ibm_brisbane
```

---

## Running Experiments

```bash
# Run all 4 experiments sequentially
python run_all_experiments.py

# Run individual experiments
python experiments/exp1_convergence.py
python experiments/exp2_arithmetic_accuracy.py
python experiments/exp3_hardware_validation.py --backend ibm_fez
python experiments/exp4_noise_sensitivity.py

# Simulate only (no IBM account required)
python run_all_experiments.py --simulate-only
```

---

## Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Convergence Analysis | Classical GD vs Hybrid QFT-GD over N≥10 trials |
| 2 | Arithmetic Accuracy | MAE of QFT subtraction vs classical subtraction |
| 3 | Hardware Validation | Real IBM hardware vs Aer simulator |
| 4 | Noise Sensitivity | Noise level sweep → arithmetic accuracy degradation |

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for a detailed system design.

---

## License

MIT License. See [LICENSE](LICENSE).
