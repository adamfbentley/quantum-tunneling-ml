# Machine Learning Classification of Quantum Tunneling Regimes

A third-year physics project investigating the use of machine learning to classify different quantum tunneling regimes in simple barrier systems, inspired by the theoretical condensed matter physics research of Michele Governale at Victoria University of Wellington.

## Project Overview

This project explores how machine learning can automatically identify different quantum tunneling behaviors from transmission probability data. We simulate electrons tunneling through various barrier configurations and train ML models to classify the underlying physics - a fundamental building block for understanding more complex quantum transport phenomena.

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

## Getting Started

1. **Prerequisites**
   - Python 3.8+
   - NumPy, SciPy, Matplotlib
   - Scikit-learn for ML
   - QuTiP for quantum simulations (optional)

2. **Installation**
   ```bash
   git clone [repository-url]
   cd quantum_dot_ml_classification
   pip install -r requirements.txt
   ```

3. **Quick Start**
   - Run `notebooks/01_physics_introduction.ipynb` for physics background
   - Execute `notebooks/02_basic_simulation.ipynb` for first simulations
   - Follow `notebooks/03_ml_classification.ipynb` for ML implementation

## Research Impact

This project has potential for:
- **Academic Publications**: ML applications in quantum transport
- **Experimental Collaborations**: Automated device characterization
- **Industry Applications**: Quantum computing and sensing technologies
- **Educational Value**: Bridging quantum physics and machine learning

## Contact & Collaboration

For questions, collaborations, or extensions of this work:
- Related to Michele Governale's research group at VUW
- Suitable for undergraduate projects through PhD research
- Open to experimental collaborations and industry partnerships

---

*This project represents an intersection of fundamental quantum physics with practical machine learning applications, offering both educational value and genuine research potential in the rapidly growing field of quantum technologies.*