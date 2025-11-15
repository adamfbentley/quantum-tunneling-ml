# Machine Learning Classification of Quantum Dot Transport Regimes

> **⚠️ PROJECT STATUS: Work in Progress**  
> This repository contains code for a quantum dot ML classification project. The core simulation code (~1,100 lines) is implemented, but notebooks and documentation are incomplete. This is an exploratory physics learning project.

A physics project exploring machine learning classification of quantum dot transport regimes, inspired by theoretical condensed matter physics research at Victoria University of Wellington.

## Project Overview

This project explores using machine learning to classify different quantum transport behaviors in quantum dot systems. The implementation includes quantum dot physics simulation and ML classification infrastructure.

## Physics Background

### Quantum Dot Transport
- Electron transport through quantum-confined systems
- Coulomb blockade and charging effects
- Quantum coherence in mesoscopic devices
- Energy-dependent conductance properties

### Transport Regimes
- **Single Quantum Dot**: Discrete energy levels and Coulomb blockade
- **Double Quantum Dot**: Coupled dots with charge/spin states
- **Quantum Dot Arrays**: Complex multi-level systems
- **Different Coupling Strengths**: Weak vs. strong coupling regimes

### Observable Signatures
- Conductance vs gate voltage curves
- Coulomb diamond patterns
- Resonance peaks and level spacing
- Temperature-dependent transport

## Research Context

This project relates to theoretical condensed matter physics research areas:
- **Quantum Transport**: Electron transport through quantum-confined nanostructures
- **Mesoscopic Physics**: Quantum coherence and charge quantization
- **Quantum Dots**: Artificial atoms with discrete energy spectra
- **Computational Methods**: Numerical simulation and ML-based classification

## Key Features

### Quantum Simulation
- Master equation approach for quantum dot dynamics
- Energy level calculations and state evolution
- Transport coefficient computation
- Temperature-dependent effects

### Machine Learning
- Classification of transport regimes
- Feature extraction from conductance data
- Model training and evaluation
- Physics-based interpretation

## Current Status

**What's Implemented:**
- ✅ Core quantum simulation code (`src/quantum_simulation/`)
- ✅ ML model infrastructure (`src/ml_models/`)
- ✅ Basic quantum dot simulator (1,100+ lines)

**What's In Progress:**
- 🚧 Jupyter notebooks (planned but not yet created)
- 🚧 Full documentation
- 🚧 Comprehensive examples
- 🚧 Test coverage

## Getting Started

**Prerequisites:**
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Scikit-learn for ML

**Installation:**
```bash
git clone https://github.com/adamfbentley/quantum-tunneling-ml.git
cd quantum-tunneling-ml
pip install -r requirements.txt
```

**Note:** Interactive notebooks are planned for future development. Current code can be explored in the `src/` directory.

## Learning Goals

This project serves as a learning exercise to explore:

- Quantum mechanics simulation (Schrödinger equation solvers)
- Feature extraction from physics data
- ML classification pipelines
- Bridging quantum physics and machine learning

## Development Roadmap

**Planned Future Work:**
- [ ] Complete interactive Jupyter notebooks
- [ ] Add comprehensive documentation
- [ ] Implement additional barrier types
- [ ] Expand ML model comparison
- [ ] Add visualization tools

## Contributing

This is a personal learning project, but suggestions and feedback are welcome via issues or pull requests.

## License

MIT License - see LICENSE file for details.