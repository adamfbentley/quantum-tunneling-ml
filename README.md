# Machine Learning Classification of Quantum Tunneling Regimes

> **⚠️ PROJECT STATUS: Work in Progress / Learning Project**  
> This repository contains initial code and design documents for a quantum tunneling ML classification project. The implementation is incomplete and notebooks are currently under development. This is a learning/exploratory project, not production-ready research.

A physics learning project exploring machine learning classification of quantum tunneling regimes in potential barrier systems, inspired by theoretical condensed matter physics research at Victoria University of Wellington.

## Project Overview

This project explores how machine learning could automatically identify different quantum tunneling behaviors from transmission probability data. The goal is to simulate electrons tunneling through various barrier configurations and train ML models to classify the underlying physics.

## Physics Background

### Quantum Tunneling Basics
- Transmission through potential barriers
- Wave-particle duality and probability amplitudes
- Energy-dependent transmission coefficients
- Different barrier shapes and their signatures

### Tunneling Regimes
- **Single Rectangular Barrier**: Classic textbook case with analytical solutions
- **Double Barrier (Resonant Tunneling)**: Shows resonance peaks and quantum interference
- **Triangular Barrier**: Represents field-emission and band-bending situations
- **Multiple Barriers**: Complex interference patterns and mini-band formation

### Observable Signatures
- Transmission probability vs energy curves
- Resonance peak positions and widths
- Oscillatory behavior from quantum interference
- Energy-dependent phase accumulation

## Research Connection

This project directly relates to Michele Governale's research areas:
- **Quantum Transport**: Understanding electron transmission through nanostructures
- **Mesoscopic Physics**: Quantum coherence in small-scale systems
- **Theoretical Foundations**: Building blocks for more complex quantum device physics
- **Computational Methods**: Bridge between analytical solutions and numerical simulations

## Implementation Plan

### Phase 1: Third-Year Assignment (Weeks 1-6)
1. **Quantum Mechanics Review** (Week 1)
   - Schrödinger equation for barrier problems
   - Transmission and reflection coefficients
   - Analytical solutions for simple cases

2. **Simulation Development** (Weeks 2-3)
   - Implement 1D Schrödinger equation solver
   - Generate transmission vs energy data for different barriers
   - Create dataset with various barrier configurations

3. **Machine Learning** (Weeks 4-5)
   - Extract features from transmission curves
   - Train classification models (3-4 simple algorithms)
   - Compare ML predictions with known physics

4. **Analysis & Report** (Week 6)
   - Interpret ML results in terms of quantum mechanics
   - Compare with analytical solutions where possible
   - Document findings and physics insights

### Key Learning Outcomes
- **Quantum Mechanics**: Deep understanding of tunneling physics
- **Computational Physics**: Numerical solution of differential equations
- **Machine Learning**: Feature extraction and classification
- **Scientific Computing**: Data analysis and visualization
- **Physics Interpretation**: Connecting ML results to underlying physics

## Expected Outcomes

### Assignment Level
- Functional quantum dot transport simulator
- Trained ML models achieving >90% classification accuracy
- Physics-based interpretation of ML features
- Comparison with traditional analysis methods

### Research Level
- Publication-quality results on ML for quantum transport
- Potential collaboration with experimental groups
- Extension to related quantum systems
- Applications to quantum technology development

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