"""
Variational Quantum Eigensolver (VQE) for H2 Molecule

This script demonstrates how to use VQE to find the ground state energy
of the hydrogen molecule (H2) at different bond distances.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

always_use_manual_expectation = True

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
        
    # Try to get Estimator from different locations
    ESTIMATOR_AVAILABLE = False
    estimator = None
    
    try:
        # Try new location (Qiskit 0.39+)
        from qiskit.primitives import StatevectorEstimator
        estimator = StatevectorEstimator()
        ESTIMATOR_AVAILABLE = True
        print("Using qiskit.primitives.StatevectorEstimator")
    except ImportError as e:
        print("Failed to import qiskit.primitives.StatevectorEstimator: {e}")
        try:
            # Try qiskit-terra location
            from qiskit_terra.primitives import Estimator
            estimator = Estimator()
            ESTIMATOR_AVAILABLE = True
            print("Using qiskit_terra.primitives.Estimator")
        except ImportError as e2:
            print("Failed to import qiskit_terra.primitives.Estimator: {e2}")
            # Fall back to manual calculation
            print("Estimator not available. Using manual expectation value calculation.")
    
    # Import visualization if available
    try:
        from qiskit.visualization import circuit_drawer
        VISUALIZATION_AVAILABLE = True
    except ImportError as e:
        print(f"Failed to import qiskit.visualization.circuit_drawer; e=<{e}>")
        VISUALIZATION_AVAILABLE = False
        
except ImportError as e:
    print(f"Import error: {e}")
    print("This script requires qiskit. Try running: pip install qiskit qiskit-aer scipy matplotlib")
    import sys
    sys.exit(1)

def get_h2_hamiltonian(distance):
    """
    Get the Hamiltonian for H2 molecule at a given distance.
    This is a simplified 2-qubit representation.
    
    Args:
        distance (float): Bond distance in Angstroms
        
    Returns:
        dict or SparsePauliOp: The Hamiltonian operator
    """
    # Coefficients for H2 in minimal basis (STO-3G)
    # These are approximate values for demonstration
    a = 0.5678
    b = -1.4508
    c = 0.6799
    d = 0.0791
    e = 0.0791
    
    # Adjust coefficients based on distance (simplified model)
    factor = np.exp(-distance)
    
    # Define Pauli operators and coefficients
    pauli_dict = {
        'II': a * factor,
        'IZ': b,
        'ZI': b,
        'ZZ': c * factor,
        'XX': d,
        'YY': e
    }
    
    # Return appropriate format
    if ESTIMATOR_AVAILABLE:
        try:
            return SparsePauliOp.from_list([(k, v) for k, v in pauli_dict.items()])
        except:
            return pauli_dict
    else:
        return pauli_dict

def create_ansatz(theta):
    """
    Create a hardware-efficient ansatz circuit.
    
    Args:
        theta (list): List of parameter values
        
    Returns:
        QuantumCircuit: Parameterized quantum circuit
    """
    circuit = QuantumCircuit(2)
    
    # First layer: RY rotations
    circuit.ry(theta[0], 0)
    circuit.ry(theta[1], 1)
    
    # Entangling layer
    circuit.cx(0, 1)
    
    # Second layer: RY rotations
    circuit.ry(theta[2], 0)
    circuit.ry(theta[3], 1)
    
    return circuit

def compute_expectation_manual(circuit, pauli_dict):
    """
    Manually compute expectation value when Estimator is not available.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit
        pauli_dict (dict): Dictionary of Pauli strings and coefficients
        
    Returns:
        float: Expectation value
    """
    expectation = 0.0
    
    for pauli_string, coeff in pauli_dict.items():
        # Create measurement circuit
        meas_circuit = circuit.copy()
        
        # Add measurement basis rotations
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                meas_circuit.h(i)
            elif pauli == 'Y':
                meas_circuit.sdg(i)
                meas_circuit.h(i)
        
        # Add measurements
        meas_circuit.measure_all()
        
        # Execute using modern Qiskit approach
        try:
            # Use Sampler for newer Qiskit versions
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
            job = sampler.run([meas_circuit], shots=8192)
            result = job.result()
            counts = result[0].data.meas.get_counts()
            
            # Calculate expectation for this Pauli term
            total = sum(counts.values())
            pauli_exp = 0
            for bitstring, count in counts.items():                # Calculate parity
                parity = 1
                for i, (bit, pauli) in enumerate(zip(bitstring[::-1], pauli_string)):
                    if pauli != 'I' and bit == '1':
                        parity *= -1
                pauli_exp += parity * count / total
            
            expectation += coeff * pauli_exp
            
        except Exception as e:
            print(f"Error in manual calculation: {e}")
            return -1.0
    
    # Ensure we return a real number
    if isinstance(expectation, complex):
        expectation = expectation.real
    return float(expectation)

def compute_expectation(theta, hamiltonian, estimator_obj):
    """
    Compute expectation value of the Hamiltonian.
    
    Args:
        theta (list): Parameter values
        hamiltonian: The Hamiltonian (SparsePauliOp or dict)
        estimator_obj: Qiskit Estimator object (can be None)
        
    Returns:
        float: Expectation value
    """
    circuit = create_ansatz(theta)
    
    if (not always_use_manual_expectation) and ESTIMATOR_AVAILABLE and estimator_obj is not None:
        try:            # Run the estimator
            result = estimator_obj.run([(circuit, hamiltonian)]).result()
            # For StatevectorEstimator, the result is stored differently
            energy = result[0].data.evs
            
            # Check if energy is an array and extract the first element if needed
            try:
                # Try to access as array first
                if hasattr(energy, '__iter__') and not isinstance(energy, (str, bytes)):
                    energy = energy[0]
            except (IndexError, TypeError):
                # If it fails, energy is already a scalar
                pass
            
            # Ensure we return a real number (expectation values should be real)
            if isinstance(energy, complex):
                energy = energy.real
            
            return float(energy)
        except Exception as e:
            print(f"Error computing expectation with Estimator: {e}")
            # Fall back to manual calculation
            if isinstance(hamiltonian, dict):
                return compute_expectation_manual(circuit, hamiltonian)
            return 0.0
    else:
        # Use manual calculation
        if isinstance(hamiltonian, dict):
            return compute_expectation_manual(circuit, hamiltonian)
        else:
            # Convert SparsePauliOp to dict if needed
            try:
                pauli_dict = {str(p): c for p, c in hamiltonian.to_list()}
                return compute_expectation_manual(circuit, pauli_dict)
            except:
                return -1.0

def vqe_h2(distance, initial_params=None):
    """
    Run VQE to find ground state energy of H2.
    
    Args:
        distance (float): Bond distance in Angstroms
        initial_params (list): Initial parameter values
        
    Returns:
        tuple: (optimal_energy, optimal_params)
    """
    # Get Hamiltonian
    hamiltonian = get_h2_hamiltonian(distance)
    
    # Initial parameters if not provided
    if initial_params is None:
        initial_params = np.random.rand(4) * 2 * np.pi
    
    # Define cost function
    def cost_function(theta):
        return compute_expectation(theta, hamiltonian, estimator)
    
    # Optimize
    result = minimize(cost_function, initial_params, method='COBYLA', 
                     options={'maxiter': 100})
    
    return result.fun, result.x

def plot_potential_energy_surface():
    """
    Plot the potential energy surface of H2 molecule.
    """
    distances = np.linspace(0.5, 3.0, 20)
    energies = []
    
    print("Computing H2 potential energy surface...")
    print("Distance (Å) | Energy (Hartree)")
    print("-" * 30)
    
    # Use same initial parameters for consistency
    params = None
    
    for d in distances:
        energy, params = vqe_h2(d, params)
        energies.append(energy)
        print(f"{d:11.2f} | {energy:15.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Bond Distance (Angstroms)', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title('H2 Molecule Potential Energy Surface (VQE)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Find and mark minimum
    min_idx = np.argmin(energies)
    plt.plot(distances[min_idx], energies[min_idx], 'r*', markersize=15)
    plt.annotate(f'Min: {distances[min_idx]:.2f} Å\nE: {energies[min_idx]:.3f}',
                xy=(distances[min_idx], energies[min_idx]),
                xytext=(distances[min_idx] + 0.3, energies[min_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('h2_vqe_energy_surface.png', dpi=300)
    print(f"\nPotential energy surface saved as 'h2_vqe_energy_surface.png'")
    plt.show()
    
    return distances, energies

if __name__ == "__main__":
    print("=" * 60)
    print("Variational Quantum Eigensolver (VQE) for H2 Molecule")
    print("=" * 60)
    
    # Run VQE for a single distance
    print("\nRunning VQE for H2 at equilibrium distance (0.74 Å)...")
    energy, params = vqe_h2(0.74)
    print(f"Ground state energy: {energy:.6f} Hartree")
    print(f"Optimal parameters: {params}")
    
    # Show the ansatz circuit
    if VISUALIZATION_AVAILABLE:
        print("\nAnsatz circuit structure:")
        ansatz = create_ansatz([Parameter('θ0'), Parameter('θ1'), 
                               Parameter('θ2'), Parameter('θ3')])
        print(ansatz)
        try:
            circuit_drawer(ansatz, output='mpl', filename='vqe_ansatz.png')
            print("Ansatz circuit saved as 'vqe_ansatz.png'")
        except Exception as e:
            print(f"Could not save circuit diagram: {e}")
    
    # Compute and plot potential energy surface
    print("\n" + "=" * 60)
    plot_potential_energy_surface()
    
    print("\n" + "=" * 60)
    print("VQE simulation completed!")
