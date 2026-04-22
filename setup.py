from setuptools import setup, find_packages

setup(
    name="hybrid_qgd",
    version="1.0.0",
    description="Hybrid Quantum-Classical Gradient Descent via QFT-Based Arithmetic",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "qiskit>=2.0.0",
        "qiskit-aer>=0.15.0",
        "qiskit-ibm-runtime>=0.20.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "pandas>=2.2.0",
        "seaborn>=0.13.0",
    ],
)
