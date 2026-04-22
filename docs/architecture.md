# Architecture: Hybrid QFT Gradient Descent

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                   Hybrid QFT Optimizer Loop                      │
│                                                                  │
│  params_t ──────────────────────────────────────────────────┐   │
│                                                              │   │
│  ┌──────────────────────┐    ┌────────────────────────────┐ │   │
│  │  Objective Circuit   │    │  Parameter Shift Rule       │ │   │
│  │  f(θ) via Qiskit     │───▶│  g_t = [f(θ+π/2)-f(θ-π/2)]│ │   │
│  │  Aer / IBM Runtime   │    │        / 2                  │ │   │
│  └──────────────────────┘    └────────────┬───────────────┘ │   │
│                                           │  gradient g_t    │   │
│                              ┌────────────▼───────────────┐ │   │
│                              │  Classical Scaling          │ │   │
│                              │  Δ_i = α · g_t[i]          │ │   │
│                              └────────────┬───────────────┘ │   │
│                                           │  delta           │   │
│                              ┌────────────▼───────────────┐ │   │
│                              │  QFT Subtraction Circuit    │ │   │
│                              │  x_{t+1} = x_t - Δ         │ │   │
│                              │  (QFT → phase rotations     │ │   │
│                              │   → IQFT → measure)         │ │   │
│                              └────────────┬───────────────┘ │   │
│                                           │  params_{t+1}    │   │
│  ◀────────────────────────────────────────┘                  │   │
└─────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `qft_arithmetic.py` | Build QFT add/subtract circuits; execute on any Qiskit backend |
| `parameter_shift.py` | Compute PSR gradients; track circuit evaluation stats |
| `hybrid_optimizer.py` | Orchestrate the full optimization loop (hybrid + classical baseline) |
| `quantum_backend.py` | Connect to Aer or IBM Runtime; handle fallback logic |
| `noise_model.py` | Build parametric noise models for Aer simulation |
| `objective_functions.py` | Provide differentiable test functions and quantum VQE loss |
| `utils.py` | Fixed-point encode/decode; MAE/RMSE helpers |

## QFT Arithmetic Circuit Depth

For n=4 qubits:
- QFT gate: depth ≈ n(n+1)/2 = 10
- Phase rotations: n = 4 single-qubit gates
- IQFT: depth ≈ 10
- Total: ~24 layers (before transpilation)

After Qiskit transpilation to native gates (e.g. ECR basis for IBM Fez):
depth increases by ~2–3× due to decomposition.

## Data Flow

```
config.yaml
    │
    ▼
BackendManager ──▶ AerSimulator (GPU/CPU)
                └▶ IBM Runtime (ibm_fez / ibm_torino)
                        │
                        ▼
QFTArithmetic.subtract(a, b, backend, shots)
    │
    ├── build_subtraction_circuit()
    │       encode_value(a) → QFT → phase_rotations(-b) → IQFT → measure
    │
    ├── generate_preset_pass_manager() → transpile
    │
    └── SamplerV2.run() → counts → decode_counts_distribution() → float
```

## Fixed-Point Encoding

Values in [v_min, v_max] are mapped to integers in [0, 2^n - 1]:

```
integer = round( (value - v_min) / (v_max - v_min) × (2^n - 1) )
```

This is basis-encoded into the quantum register via X gates (LSB first).
After measurement, the integer is decoded back to float. The encoding
introduces a quantization error bounded by 1 LSB = (v_max - v_min) / (2^n - 1).

For n=4: LSB = 2/15 ≈ 0.133
For n=6: LSB = 2/63 ≈ 0.032
For n=8: LSB = 2/255 ≈ 0.008
