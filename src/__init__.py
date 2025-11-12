# Quantum Dot ML Classification Package
"""
Machine Learning Classification of Quantum Dot Spin States

A research project investigating the use of machine learning to classify
quantum dot spin states from transport measurements, inspired by Michele
Governale's theoretical condensed matter physics research at Victoria
University of Wellington.
"""

__version__ = "1.0.0"
__author__ = "Adam Bentley"
__institution__ = "Victoria University of Wellington"
__research_group__ = "Michele Governale's Theoretical Condensed Matter Physics"

# Import main modules
from .quantum_simulation import QuantumDotSimulator, extract_features, create_balanced_dataset
from .ml_models import QuantumDotMLClassifier, run_complete_analysis

__all__ = [
    'QuantumDotSimulator',
    'extract_features', 
    'create_balanced_dataset',
    'QuantumDotMLClassifier',
    'run_complete_analysis'
]