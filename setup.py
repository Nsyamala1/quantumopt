from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="quantumopt",
    version="0.1.0",
    author="Naveen Syamala",
    author_email="your@email.com",
    description="AI-driven quantum circuit compiler using GNN + LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nsyamala1/quantumopt",
    packages=find_packages(),
    package_data={
        "quantumopt": [
            "models/weights/*.pt",
            "models/weights/*.txt",
            "models/weights/*.json",
        ]
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "qiskit>=1.0.0",
        "qiskit-aer>=0.13.0",
        "qiskit-ibm-runtime>=0.20.0",
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "llm": ["anthropic>=0.20.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "quantum computing",
        "quantum compiler",
        "circuit optimization",
        "graph neural network",
        "qiskit",
        "ibm quantum",
        "variational quantum eigensolver",
        "QAOA",
    ],
)
