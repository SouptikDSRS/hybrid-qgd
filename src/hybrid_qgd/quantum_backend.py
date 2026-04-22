"""
quantum_backend.py
------------------
Unified backend manager for the hybrid QGD framework.

Handles
-------
- Aer GPU/CPU simulator selection
- IBM Quantum Runtime connection and backend selection
- Fallback logic (hardware unavailable → simulator)
- Backend metadata reporting

Usage
-----
    from hybrid_qgd.quantum_backend import BackendManager

    mgr = BackendManager("configs/config.yaml", "configs/ibm_credentials.yaml")
    backend = mgr.get_backend("simulation")   # Aer simulator
    backend = mgr.get_backend("hardware")     # IBM hardware (with fallback)
"""

from __future__ import annotations

import os
import yaml
import warnings
from pathlib import Path
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Backend Manager
# ─────────────────────────────────────────────────────────────────────────────

class BackendManager:
    """Manages Aer and IBM Quantum backends.

    Parameters
    ----------
    config_path      : path to config.yaml
    credentials_path : path to ibm_credentials.yaml (optional; set to None
                       to skip IBM Runtime initialisation)
    """

    def __init__(self,
                 config_path: str = "configs/config.yaml",
                 credentials_path: str | None = "configs/ibm_credentials.yaml"):
        self._config    = self._load_yaml(config_path)
        self._creds     = self._load_yaml(credentials_path) if credentials_path else {}
        self._ibm_service = None
        self._aer_backend = None

    # ── YAML loader ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p) as fh:
            return yaml.safe_load(fh) or {}

    # ── Simulator ─────────────────────────────────────────────────────────────

    def get_simulator(self) -> object:
        """Return a configured Aer simulator backend.

        Respects config.yaml:
          backends.simulation.method   (statevector | density_matrix | automatic)
          backends.simulation.gpu      (true | false)
        """
        if self._aer_backend is not None:
            return self._aer_backend

        from qiskit_aer import AerSimulator

        sim_cfg = self._config.get("backends", {}).get("simulation", {})
        method  = sim_cfg.get("method", "automatic")
        use_gpu = sim_cfg.get("gpu", False)

        device  = "GPU" if use_gpu else "CPU"
        try:
            backend = AerSimulator(method=method, device=device)
        except Exception:
            # GPU not available — fall back to CPU
            backend = AerSimulator(method=method)

        self._aer_backend = backend
        return backend

    # ── IBM Runtime ───────────────────────────────────────────────────────────

    def _connect_ibm(self) -> None:
        """Initialise IBM Quantum Runtime service (once)."""
        if self._ibm_service is not None:
            return

        from qiskit_ibm_runtime import QiskitRuntimeService

        try:
            # ✅ Load saved credentials automatically
            self._ibm_service = QiskitRuntimeService()

            print("[BackendManager] IBM Quantum service initialized successfully")

        except Exception as e:
            raise RuntimeError(
                "IBM Quantum connection failed.\n"
                "Make sure you have saved your token using:\n"
                "QiskitRuntimeService.save_account(token='YOUR_TOKEN')"
            ) from e

    def get_hardware_backend(self) -> object:
        """Return the least-busy preferred IBM hardware backend.

        Falls back to simulator if config.backends.hardware.fallback_to_sim
        is true and no hardware backend is available.

        Returns
        -------
        Qiskit backend object
        """
        try:
            self._connect_ibm()
        except RuntimeError as exc:
            hw_cfg = self._config.get("backends", {}).get("hardware", {})
            if hw_cfg.get("fallback_to_sim", True):
                warnings.warn(
                    f"IBM connection failed ({exc}). Falling back to simulator.",
                    stacklevel=2,
                )
                return self.get_simulator()
            raise

        preferred = self._creds.get(
            "preferred_backends",
            self._config.get("backends", {}).get("hardware", {}).get("preferred", []),
        )
        opt_level = (self._config
                     .get("backends", {})
                     .get("hardware", {})
                     .get("optimization_level", 1))

        # Try preferred backends in order
        for name in preferred:
            try:
                backend = self._ibm_service.backend(name)
                print(f"[BackendManager] Connected to IBM backend: {name}")
                return backend
            except Exception:
                continue

        # Fall back to least-busy available backend
        try:
            backend = self._ibm_service.least_busy(
                operational=True, simulator=False, min_num_qubits=5
            )
            print(f"[BackendManager] Using least-busy backend: {backend.name}")
            return backend
        except Exception as exc:
            hw_cfg = self._config.get("backends", {}).get("hardware", {})
            if hw_cfg.get("fallback_to_sim", True):
                warnings.warn(
                    f"No hardware backend available ({exc}). Falling back to simulator.",
                    stacklevel=2,
                )
                return self.get_simulator()
            raise

    # ── Unified entry point ───────────────────────────────────────────────────

    def get_backend(self,
                    mode: Literal["simulation", "hardware"] = "simulation") -> object:
        """Return the appropriate backend for the given mode.

        Parameters
        ----------
        mode : "simulation" → Aer ; "hardware" → IBM (with fallback)
        """
        if mode == "simulation":
            return self.get_simulator()
        elif mode == "hardware":
            return self.get_hardware_backend()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'simulation' or 'hardware'.")

    # ── Metadata ──────────────────────────────────────────────────────────────

    def backend_info(self, backend) -> dict:
        """Return metadata dict for a backend."""
        info: dict = {"name": getattr(backend, "name", str(backend))}
        try:
            props = backend.properties()
            if props:
                info["n_qubits"] = len(props.qubits)
                info["t1_avg_us"] = float(
                    sum(props.t1(i) for i in range(len(props.qubits)))
                    / len(props.qubits)
                ) * 1e6
        except Exception:
            pass
        return info

    @property
    def config(self) -> dict:
        return self._config

    @property
    def qft_config(self) -> dict:
        return self._config.get("qft_arithmetic", {})

    @property
    def optimizer_config(self) -> dict:
        return self._config.get("optimizer", {})
