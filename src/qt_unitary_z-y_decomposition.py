"""
Unitary Z-Y Decomposition for Single-Qubit Gates

This script demonstrates how to decompose arbitrary single-qubit unitary matrices
into a sequence of Z and Y rotations (ZYZ decomposition).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.visualization import plot_bloch_multivector
    QISKIT_AVAILABLE = True
    print("Qiskit available for circuit construction and visualization")
except ImportError as e:
    print(f"Qiskit not available: {e}")
    QISKIT_AVAILABLE = False

def pauli_matrices():
    """Return the Pauli matrices."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

def rotation_matrices(theta, phi, lam):
    """
    Generate rotation matrices for ZYZ decomposition.
    
    Args:
        theta (float): Y rotation angle
        phi (float): First Z rotation angle
        lam (float): Second Z rotation angle
    
    Returns:
        tuple: (Rz_phi, Ry_theta, Rz_lam) rotation matrices
    """
    # Z rotation matrix
    def Rz(angle):
        return np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)
    
    # Y rotation matrix
    def Ry(angle):
        return np.array([
            [np.cos(angle / 2), -np.sin(angle / 2)],
            [np.sin(angle / 2), np.cos(angle / 2)]
        ], dtype=complex)
    
    return Rz(phi), Ry(theta), Rz(lam)

def zyz_decomposition(U):
    """
    Decompose a single-qubit unitary matrix into ZYZ rotations.
    
    U = e^(iα) * Rz(φ) * Ry(θ) * Rz(λ)
    
    Args:
        U (np.ndarray): 2x2 unitary matrix
    
    Returns:
        tuple: (alpha, phi, theta, lambda) angles in radians
    """
    if U.shape != (2, 2):
        raise ValueError("Input must be a 2x2 matrix")
    
    # Normalize to ensure det(U) = 1 (special unitary)
    det_U = np.linalg.det(U)
    # det(U) = det(e^(iα) × U_SU(2)) = e^(iα)^2 × det(U_SU(2)) = e^(i×2α) × 1 = e^(i×2α)
    alpha = np.angle(det_U) / 2
    U_su2 = U / np.sqrt(det_U)
    
    # Extract angles from the SU(2) matrix
    # U_su2 = [[a, b], [-b*, a*]] for some complex a, b with |a|^2 + |b|^2 = 1
    a = U_su2[0, 0]
    b = U_su2[0, 1]
    
    # Calculate theta from |b|
    theta = 2 * np.arcsin(min(abs(b), 1.0))  # Clamp to avoid numerical errors
    
    if abs(theta) < 1e-10:  # theta ≈ 0
        # U is approximately diagonal
        phi = 0
        # lam = 2 * np.angle(a)
        lam = 2 * np.angle(U_su2[1, 1])
    elif abs(theta - np.pi) < 1e-10:  # theta ≈ π
        # Special case where sin(theta/2) ≈ 1
        # phi = 2 * np.angle(b)
        phi = 2 * np.angle(U_su2[1, 0])
        lam = 0
    else:
        # General case
        # phi = np.angle(-U_su2[1, 0]) - np.angle(U_su2[0, 1])
        # lam = np.angle(U_su2[1, 0]) + np.angle(U_su2[0, 1])
        phi = np.angle(U_su2[1, 0] * U_su2[1, 1])
        lam = np.angle(-U_su2[0, 1] * U_su2[1, 1])
    
    return alpha, phi, theta, lam

def reconstruct_unitary(alpha, phi, theta, lam):
    """
    Reconstruct the unitary matrix from ZYZ angles.
    
    Args:
        alpha (float): Global phase
        phi (float): First Z rotation
        theta (float): Y rotation
        lam (float): Second Z rotation
    
    Returns:
        np.ndarray: Reconstructed 2x2 unitary matrix
    """
    Rz_phi, Ry_theta, Rz_lam = rotation_matrices(theta, phi, lam)
    U_reconstructed = np.exp(1j * alpha) * Rz_phi @ Ry_theta @ Rz_lam
    return U_reconstructed

def verify_decomposition(U, alpha, phi, theta, lam, tolerance=1e-10):
    """
    Verify that the decomposition is correct.
    
    Args:
        U (np.ndarray): Original unitary matrix
        alpha, phi, theta, lam (float): Decomposition angles
        tolerance (float): Numerical tolerance
    
    Returns:
        tuple: (is_correct, error_norm)
    """
    U_reconstructed = reconstruct_unitary(alpha, phi, theta, lam)
    error = np.linalg.norm(U - U_reconstructed)
    return error < tolerance, error

def create_qiskit_circuit(phi, theta, lam):
    """
    Create a Qiskit circuit implementing the ZYZ decomposition.
    
    Args:
        phi (float): First Z rotation
        theta (float): Y rotation
        lam (float): Second Z rotation
    
    Returns:
        QuantumCircuit: Circuit implementing the decomposition
    """
    if not QISKIT_AVAILABLE:
        print("Qiskit not available for circuit creation")
        return None
    
    circuit = QuantumCircuit(1)
    circuit.rz(phi, 0)
    circuit.ry(theta, 0)
    circuit.rz(lam, 0)
    return circuit

def demo_common_gates():
    """Demonstrate ZYZ decomposition for common quantum gates."""
    print("ZYZ Decomposition of Common Quantum Gates")
    print("=" * 50)
    
    # Define common gates
    I, X, Y, Z = pauli_matrices()
    
    gates = {
        "Identity": I,
        "Pauli-X": X,
        "Pauli-Y": Y,
        "Pauli-Z": Z,
        "Hadamard": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        "S-gate": np.array([[1, 0], [0, 1j]], dtype=complex),
        "T-gate": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    }
    
    for name, gate in gates.items():
        print(f"\n{name} Gate:")
        alpha, phi, theta, lam = zyz_decomposition(gate)
        
        print(f"  α = {alpha:.4f} rad ({np.degrees(alpha):.2f}°)")
        print(f"  φ = {phi:.4f} rad ({np.degrees(phi):.2f}°)")
        print(f"  θ = {theta:.4f} rad ({np.degrees(theta):.2f}°)")
        print(f"  λ = {lam:.4f} rad ({np.degrees(lam):.2f}°)")
        
        # Verify decomposition
        is_correct, error = verify_decomposition(gate, alpha, phi, theta, lam)
        print(f"  Verification: {'✓' if is_correct else '✗'} (error: {error:.2e})")
        
        # Show Qiskit circuit if available
        if QISKIT_AVAILABLE:
            circuit = create_qiskit_circuit(phi, theta, lam)
            print(f"  Qiskit circuit: {circuit.size()} gates")

def plot_decomposition_visualization():
    """Create visualization of the decomposition process."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate random unitary matrices and decompose them
    np.random.seed(42)
    n_samples = 50
    
    angles_data = {'alpha': [], 'phi': [], 'theta': [], 'lambda': []}
    errors = []
    
    for _ in range(n_samples):
        # Generate random 2x2 unitary matrix
        # Method: QR decomposition of random complex matrix
        A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        Q, R = np.linalg.qr(A)
        # Ensure det(Q) = 1
        Q = Q / (np.linalg.det(Q) ** 0.5)
        
        # Decompose
        alpha, phi, theta, lam = zyz_decomposition(Q)
        angles_data['alpha'].append(alpha)
        angles_data['phi'].append(phi)
        angles_data['theta'].append(theta)
        angles_data['lambda'].append(lam)
        
        # Calculate reconstruction error
        _, error = verify_decomposition(Q, alpha, phi, theta, lam)
        errors.append(error)
    
    # Plot angle distributions
    angle_names = ['α (global phase)', 'φ (first Z)', 'θ (Y rotation)', 'λ (second Z)']
    for i, (key, name) in enumerate(zip(angles_data.keys(), angle_names)):
        ax = axes[i // 2, i % 2]
        ax.hist(angles_data[key], bins=15, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{name} (radians)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zyz_decomposition_analysis.png', dpi=300, bbox_inches='tight')
    print("\nAngle distribution plot saved as 'zyz_decomposition_analysis.png'")
    plt.show()
    
    # Print statistics
    print(f"\nDecomposition Statistics (n={n_samples}):")
    print(f"Mean reconstruction error: {np.mean(errors):.2e}")
    print(f"Max reconstruction error: {np.max(errors):.2e}")
    print(f"All decompositions accurate: {'Yes' if np.max(errors) < 1e-10 else 'No'}")

def interactive_decomposition():
    """Interactive function to decompose user-specified unitaries."""
    print("\nInteractive Unitary Decomposition")
    print("-" * 35)
    
    # Example: Rotation around arbitrary axis
    print("Example: Rotation around arbitrary axis (x̂ + ŷ + ẑ)/√3")
    
    # Normalized rotation axis
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    angle = np.pi / 3  # 60 degrees
    
    # Rotation matrix using Rodrigues' formula
    # R = I + sin(θ) * K + (1 - cos(θ)) * K²
    # where K is the skew-symmetric matrix of the axis
    I, X, Y, Z = pauli_matrices()
    
    # For single qubit: U = cos(θ/2) * I - i * sin(θ/2) * (n·σ)
    sigma_n = axis[0] * X + axis[1] * Y + axis[2] * Z
    U = np.cos(angle/2) * I - 1j * np.sin(angle/2) * sigma_n
    
    print(f"Rotation angle: {angle:.4f} rad ({np.degrees(angle):.1f}°)")
    print(f"Rotation axis: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
    
    # Decompose
    alpha, phi, theta, lam = zyz_decomposition(U)
    
    print(f"\nZYZ Decomposition:")
    print(f"α = {alpha:.4f} rad ({np.degrees(alpha):.2f}°)")
    print(f"φ = {phi:.4f} rad ({np.degrees(phi):.2f}°)")
    print(f"θ = {theta:.4f} rad ({np.degrees(theta):.2f}°)")
    print(f"λ = {lam:.4f} rad ({np.degrees(lam):.2f}°)")
    
    # Verify
    is_correct, error = verify_decomposition(U, alpha, phi, theta, lam)
    print(f"\nVerification: {'✓' if is_correct else '✗'} (error: {error:.2e})")
    
    if QISKIT_AVAILABLE:
        circuit = create_qiskit_circuit(phi, theta, lam)
        print(f"\nQiskit Circuit:")
        print(circuit)

if __name__ == "__main__":
    print("=" * 60)
    print("Unitary Z-Y Decomposition for Single-Qubit Gates")
    print("=" * 60)
    
    # Demonstrate decomposition of common gates
    demo_common_gates()
    
    # Create visualization
    plot_decomposition_visualization()
    
    # Interactive example
    interactive_decomposition()
    
    print("\n" + "=" * 60)
    print("ZYZ Decomposition demonstration completed!")
    print("\nKey takeaways:")
    print("• Any single-qubit unitary can be decomposed as U = e^(iα) Rz(φ) Ry(θ) Rz(λ)")
    print("• This provides a universal gate set for single-qubit operations")
    print("• The decomposition is unique up to multiples of 2π in the angles")
    print("• Useful for compiling arbitrary rotations to hardware-native gates")
