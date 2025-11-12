"""
Quantum Dot Transport Simulation
================================

This module implements quantum dot physics for machine learning classification
of spin states based on transport measurements.

Author: Adam Bentley
Institution: Victoria University of Wellington
Connection: Michele Governale's research group
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from numba import jit
import warnings
warnings.filterwarnings('ignore')

class QuantumDotSimulator:
    """
    Simulates quantum dot transport properties for machine learning applications.
    
    This class implements the essential physics of quantum dots including:
    - Coulomb blockade and charging energy
    - Zeeman splitting in magnetic field
    - Spin-orbit coupling effects
    - Sequential tunneling transport
    """
    
    def __init__(self, n_levels=4, charging_energy=1.0, g_factor=2.0):
        """
        Initialize quantum dot simulator.
        
        Parameters:
        -----------
        n_levels : int
            Number of single-particle energy levels
        charging_energy : float
            Coulomb charging energy in meV
        g_factor : float
            Landé g-factor for Zeeman splitting
        """
        self.n_levels = n_levels
        self.E_c = charging_energy  # Charging energy
        self.g = g_factor
        self.mu_B = 0.057  # Bohr magneton in meV/T
        
        # Single-particle energy levels (equally spaced for simplicity)
        self.epsilon = np.linspace(0, 2.0, n_levels)
        
    def many_body_spectrum(self, N_electrons, B_field=0.0, spin_orbit=0.0):
        """
        Calculate many-body energy spectrum for N electrons.
        
        Parameters:
        -----------
        N_electrons : int
            Number of electrons in the dot
        B_field : float
            Magnetic field in Tesla
        spin_orbit : float
            Spin-orbit coupling strength in meV
            
        Returns:
        --------
        energies : array
            Many-body energy eigenvalues
        spin_states : array
            Total spin quantum numbers
        """
        if N_electrons == 0:
            return np.array([0.0]), np.array([0.0])
        
        # Zeeman energy
        E_Z = 0.5 * self.g * self.mu_B * B_field
        
        # Generate all possible configurations
        configs = self._generate_configurations(N_electrons)
        
        energies = []
        spins = []
        
        for config in configs:
            # Single-particle energies
            E_sp = sum(self.epsilon[i//2] for i in config)
            
            # Zeeman energy (spin up: +1/2, spin down: -1/2)
            E_zeeman = sum(E_Z if i%2==0 else -E_Z for i in config)
            
            # Coulomb interaction energy
            E_coulomb = 0.5 * self.E_c * N_electrons * (N_electrons - 1)
            
            # Spin-orbit coupling (simplified)
            E_so = spin_orbit * sum((-1)**(i%2) * (i//2) for i in config)
            
            total_energy = E_sp + E_zeeman + E_coulomb + E_so
            
            # Calculate total spin
            n_up = sum(1 for i in config if i%2==0)
            n_down = N_electrons - n_up
            total_spin = abs(n_up - n_down) / 2.0
            
            energies.append(total_energy)
            spins.append(total_spin)
            
        return np.array(energies), np.array(spins)
    
    def _generate_configurations(self, N_electrons):
        """Generate all possible electron configurations."""
        from itertools import combinations
        
        # Each orbital can hold 2 electrons (spin up=0, spin down=1)
        # For n_levels orbitals, we have 2*n_levels single-particle states
        max_states = 2 * self.n_levels
        
        # Generate all combinations of N_electrons from available states
        all_configs = list(combinations(range(max_states), N_electrons))
        
        return all_configs
    
    def ground_state_properties(self, N_electrons, B_field=0.0, spin_orbit=0.0):
        """
        Calculate ground state energy and spin.
        
        Returns:
        --------
        E_ground : float
            Ground state energy
        S_ground : float
            Ground state total spin
        """
        energies, spins = self.many_body_spectrum(N_electrons, B_field, spin_orbit)
        
        if len(energies) == 0:
            return 0.0, 0.0
            
        ground_idx = np.argmin(energies)
        return energies[ground_idx], spins[ground_idx]
    
    def addition_energy(self, N_electrons, B_field=0.0, spin_orbit=0.0):
        """
        Calculate electron addition energy: E(N+1) + E(N-1) - 2*E(N)
        
        This quantity is measurable in transport experiments.
        """
        if N_electrons == 0:
            E_0 = 0.0
            E_1, _ = self.ground_state_properties(1, B_field, spin_orbit)
            return E_1
        
        E_N_minus_1, _ = self.ground_state_properties(N_electrons-1, B_field, spin_orbit)
        E_N, _ = self.ground_state_properties(N_electrons, B_field, spin_orbit)
        E_N_plus_1, _ = self.ground_state_properties(N_electrons+1, B_field, spin_orbit)
        
        return E_N_plus_1 + E_N_minus_1 - 2*E_N
    
    def conductance_peak(self, N_electrons, V_gate_range, B_field=0.0, 
                        spin_orbit=0.0, temperature=0.1, tunnel_rate=0.1):
        """
        Calculate conductance vs gate voltage for Coulomb blockade peak.
        
        Parameters:
        -----------
        N_electrons : int
            Base number of electrons
        V_gate_range : array
            Gate voltage range in meV
        temperature : float
            Temperature in meV (k_B * T)
        tunnel_rate : float
            Tunnel coupling strength
            
        Returns:
        --------
        conductance : array
            Differential conductance vs gate voltage
        """
        conductance = np.zeros_like(V_gate_range)
        
        # Ground state energies
        E_N, S_N = self.ground_state_properties(N_electrons, B_field, spin_orbit)
        E_N_plus_1, S_N_plus_1 = self.ground_state_properties(N_electrons+1, B_field, spin_orbit)
        
        # Addition energy
        mu_N = E_N_plus_1 - E_N
        
        for i, V_g in enumerate(V_gate_range):
            # Chemical potential relative to addition energy
            delta_mu = V_g - mu_N
            
            # Thermal broadening
            if temperature > 0:
                # Fermi function derivative for peak shape
                x = delta_mu / temperature
                if abs(x) < 50:  # Avoid overflow
                    fermi_deriv = np.exp(x) / (temperature * (1 + np.exp(x))**2)
                else:
                    fermi_deriv = 0.0
            else:
                # Delta function limit
                fermi_deriv = 1.0 if abs(delta_mu) < 0.01 else 0.0
            
            # Conductance proportional to tunnel rate and thermal broadening
            conductance[i] = tunnel_rate * fermi_deriv
            
        return conductance
    
    def generate_transport_data(self, max_electrons=6, B_field_range=None, 
                               V_gate_range=None, noise_level=0.0):
        """
        Generate synthetic transport dataset for machine learning.
        
        Parameters:
        -----------
        max_electrons : int
            Maximum number of electrons to consider
        B_field_range : array
            Magnetic field values to sample
        V_gate_range : array
            Gate voltage range for each measurement
        noise_level : float
            Gaussian noise level relative to signal
            
        Returns:
        --------
        dataset : dict
            Dictionary containing:
            - 'conductance': conductance traces
            - 'labels': spin state labels
            - 'parameters': physical parameters
            - 'metadata': measurement conditions
        """
        if B_field_range is None:
            B_field_range = np.linspace(0, 2.0, 5)  # 0 to 2 Tesla
        
        if V_gate_range is None:
            V_gate_range = np.linspace(-5.0, 5.0, 100)  # Gate voltage in meV
        
        dataset = {
            'conductance': [],
            'labels': [],
            'parameters': [],
            'metadata': []
        }
        
        sample_id = 0
        
        for N in range(1, max_electrons + 1):
            for B in B_field_range:
                # Calculate ground state spin
                _, S_ground = self.ground_state_properties(N, B)
                
                # Generate conductance trace
                G = self.conductance_peak(N-1, V_gate_range, B)
                
                # Add noise if specified
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * np.max(G), len(G))
                    G += noise
                
                # Classify spin state
                spin_label = int(2 * S_ground)  # 0=singlet, 1=doublet, 2=triplet, etc.
                
                dataset['conductance'].append(G)
                dataset['labels'].append(spin_label)
                dataset['parameters'].append({
                    'N_electrons': N,
                    'B_field': B,
                    'total_spin': S_ground,
                    'charging_energy': self.E_c,
                    'g_factor': self.g
                })
                dataset['metadata'].append({
                    'sample_id': sample_id,
                    'V_gate_range': V_gate_range.copy(),
                    'noise_level': noise_level
                })
                
                sample_id += 1
        
        # Convert to numpy arrays
        dataset['conductance'] = np.array(dataset['conductance'])
        dataset['labels'] = np.array(dataset['labels'])
        
        return dataset
    
    def visualize_spectrum(self, max_electrons=4, B_field=1.0):
        """
        Visualize many-body energy spectrum vs electron number.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy spectrum
        for N in range(max_electrons + 1):
            energies, spins = self.many_body_spectrum(N, B_field)
            
            for E, S in zip(energies, spins):
                color = ['blue', 'red', 'green', 'purple', 'orange'][int(2*S)]
                ax1.scatter(N, E, c=color, s=50, alpha=0.7)
        
        ax1.set_xlabel('Number of Electrons')
        ax1.set_ylabel('Energy (meV)')
        ax1.set_title(f'Many-Body Energy Spectrum (B = {B_field} T)')
        ax1.grid(True, alpha=0.3)
        
        # Addition energy
        N_range = range(1, max_electrons + 1)
        add_energies = [self.addition_energy(N, B_field) for N in N_range]
        
        ax2.plot(N_range, add_energies, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Electrons')
        ax2.set_ylabel('Addition Energy (meV)')
        ax2.set_title('Electron Addition Energy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_conductance(self, N_electrons=3, B_fields=[0.0, 0.5, 1.0, 2.0]):
        """
        Visualize conductance peaks for different magnetic fields.
        """
        V_gate = np.linspace(-8, 8, 200)
        
        plt.figure(figsize=(10, 6))
        
        for B in B_fields:
            G = self.conductance_peak(N_electrons-1, V_gate, B)
            plt.plot(V_gate, G, label=f'B = {B} T', linewidth=2)
        
        plt.xlabel('Gate Voltage (meV)')
        plt.ylabel('Conductance (e²/h)')
        plt.title(f'Coulomb Blockade Peak (N = {N_electrons-1} → {N_electrons})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Utility functions for machine learning pipeline

def extract_features(conductance_data):
    """
    Extract relevant features from conductance traces for ML classification.
    
    Parameters:
    -----------
    conductance_data : array
        Array of conductance traces
        
    Returns:
    --------
    features : array
        Feature matrix for ML algorithms
    feature_names : list
        Names of extracted features
    """
    n_samples, n_points = conductance_data.shape
    n_features = 12  # Number of features to extract
    
    features = np.zeros((n_samples, n_features))
    feature_names = [
        'peak_height', 'peak_position', 'peak_width', 'peak_asymmetry',
        'total_area', 'baseline_level', 'noise_level', 'peak_sharpness',
        'left_tail_slope', 'right_tail_slope', 'second_moment', 'skewness'
    ]
    
    for i, trace in enumerate(conductance_data):
        # Peak height
        features[i, 0] = np.max(trace)
        
        # Peak position
        features[i, 1] = np.argmax(trace)
        
        # Peak width (full width at half maximum)
        half_max = features[i, 0] / 2
        indices = np.where(trace >= half_max)[0]
        if len(indices) > 0:
            features[i, 2] = indices[-1] - indices[0]
        else:
            features[i, 2] = 0
        
        # Peak asymmetry
        peak_pos = int(features[i, 1])
        if peak_pos > 0 and peak_pos < len(trace) - 1:
            left_area = np.sum(trace[:peak_pos])
            right_area = np.sum(trace[peak_pos:])
            total_area = left_area + right_area
            if total_area > 0:
                features[i, 3] = (right_area - left_area) / total_area
            else:
                features[i, 3] = 0
        else:
            features[i, 3] = 0
        
        # Total area under curve
        features[i, 4] = np.trapz(trace)
        
        # Baseline level (average of first and last 10 points)
        features[i, 5] = (np.mean(trace[:10]) + np.mean(trace[-10:])) / 2
        
        # Noise level (standard deviation of baseline)
        baseline_points = np.concatenate([trace[:10], trace[-10:]])
        features[i, 6] = np.std(baseline_points)
        
        # Peak sharpness (second derivative at peak)
        if peak_pos > 1 and peak_pos < len(trace) - 2:
            second_deriv = trace[peak_pos-1] - 2*trace[peak_pos] + trace[peak_pos+1]
            features[i, 7] = abs(second_deriv)
        else:
            features[i, 7] = 0
        
        # Tail slopes
        peak_pos = int(features[i, 1])
        if peak_pos > 10:
            left_slope = (trace[peak_pos] - trace[peak_pos-10]) / 10
            features[i, 8] = left_slope
        else:
            features[i, 8] = 0
            
        if peak_pos < len(trace) - 10:
            right_slope = (trace[peak_pos+10] - trace[peak_pos]) / 10
            features[i, 9] = right_slope
        else:
            features[i, 9] = 0
        
        # Statistical moments
        x = np.arange(len(trace))
        if np.sum(trace) > 0:
            # Second moment (variance)
            mean_x = np.average(x, weights=trace)
            features[i, 10] = np.average((x - mean_x)**2, weights=trace)
            
            # Skewness
            if features[i, 10] > 0:
                features[i, 11] = np.average((x - mean_x)**3, weights=trace) / (features[i, 10]**1.5)
            else:
                features[i, 11] = 0
        else:
            features[i, 10] = 0
            features[i, 11] = 0
    
    return features, feature_names


def create_balanced_dataset(simulator, samples_per_class=100, max_electrons=6):
    """
    Create a balanced dataset for machine learning with equal samples per spin class.
    
    Parameters:
    -----------
    simulator : QuantumDotSimulator
        Initialized simulator object
    samples_per_class : int
        Number of samples to generate for each spin class
    max_electrons : int
        Maximum number of electrons to consider
        
    Returns:
    --------
    balanced_data : dict
        Balanced dataset ready for ML training
    """
    # Generate full dataset
    full_data = simulator.generate_transport_data(max_electrons=max_electrons)
    
    # Count samples per class
    unique_labels, counts = np.unique(full_data['labels'], return_counts=True)
    print(f"Original class distribution: {dict(zip(unique_labels, counts))}")
    
    # Create balanced dataset
    balanced_data = {
        'conductance': [],
        'labels': [],
        'parameters': [],
        'metadata': []
    }
    
    for label in unique_labels:
        # Find indices for this label
        label_indices = np.where(full_data['labels'] == label)[0]
        
        # Sample with replacement if needed
        if len(label_indices) >= samples_per_class:
            selected_indices = np.random.choice(label_indices, samples_per_class, replace=False)
        else:
            selected_indices = np.random.choice(label_indices, samples_per_class, replace=True)
        
        # Add selected samples to balanced dataset
        for idx in selected_indices:
            balanced_data['conductance'].append(full_data['conductance'][idx])
            balanced_data['labels'].append(full_data['labels'][idx])
            balanced_data['parameters'].append(full_data['parameters'][idx])
            balanced_data['metadata'].append(full_data['metadata'][idx])
    
    # Convert to numpy arrays
    balanced_data['conductance'] = np.array(balanced_data['conductance'])
    balanced_data['labels'] = np.array(balanced_data['labels'])
    
    # Verify balance
    unique_labels, counts = np.unique(balanced_data['labels'], return_counts=True)
    print(f"Balanced class distribution: {dict(zip(unique_labels, counts))}")
    
    return balanced_data


if __name__ == "__main__":
    # Demonstration of quantum dot simulator
    print("Quantum Dot Transport Simulation")
    print("=" * 40)
    
    # Create simulator
    simulator = QuantumDotSimulator(n_levels=3, charging_energy=2.0)
    
    # Visualize physics
    print("Generating energy spectrum visualization...")
    simulator.visualize_spectrum(max_electrons=4, B_field=1.0)
    
    print("Generating conductance peak visualization...")
    simulator.visualize_conductance(N_electrons=3)
    
    # Generate dataset
    print("Generating transport dataset...")
    dataset = simulator.generate_transport_data(max_electrons=4)
    
    print(f"Generated {len(dataset['conductance'])} samples")
    print(f"Conductance trace shape: {dataset['conductance'][0].shape}")
    print(f"Unique spin labels: {np.unique(dataset['labels'])}")
    
    # Extract features
    print("Extracting features for machine learning...")
    features, feature_names = extract_features(dataset['conductance'])
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features: {feature_names}")
    
    print("\nQuantum dot simulation module ready for ML classification!")