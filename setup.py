"""
setup.py — QuantumOpt package installer.

Install in development mode:
    pip install -e .

Install from PyPI (future):
    pip install quantumopt
"""

from setuptools import setup, find_packages
from pathlib import Path


def read_requirements():
    """Read requirements from requirements.txt."""
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        return [
            line.strip()
            for line in req_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []


def read_readme():
    """Read long description from README.md."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="quantumopt",
    version="0.1.0",
    author="QuantumOpt Team",
    description="AI-driven quantum circuit compiler using GNN + Claude LLM for IBM Quantum hardware",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/quantumopt/quantumopt",
    packages=find_packages(exclude=["tests*", "benchmarks*"]),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="quantum computing, circuit optimization, GNN, compiler, qiskit",
    entry_points={
        "console_scripts": [
            "quantumopt=quantumopt.compiler:main",
        ],
    },
    include_package_data=True,
    package_data={
        "quantumopt": ["models/weights/*.pt"],
    },
)
