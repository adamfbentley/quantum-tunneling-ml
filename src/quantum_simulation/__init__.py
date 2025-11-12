# Quantum Simulation Module
"""
Quantum dot physics simulation for machine learning applications.

This module provides tools for simulating quantum dot transport properties
and generating synthetic datasets for machine learning classification.
"""

from .quantum_dot_simulator import QuantumDotSimulator, extract_features, create_balanced_dataset

__all__ = ['QuantumDotSimulator', 'extract_features', 'create_balanced_dataset']