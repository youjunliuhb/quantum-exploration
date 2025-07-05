"""
COBYLA Optimizer for Quantum Variational Algorithms

This script demonstrates the COBYLA optimization algorithm applied to
quantum circuits, with visualization of the optimization landscape and convergence.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector
    
    # Try to get simulator
    try:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(method='statevector')
        print("Using AerSimulator")
    except ImportError:
        try:
            from qiskit.providers.aer import AerSimulator
            simulator = AerSimulator(method='statevector')
        except ImportError:
            from qiskit.providers.basicaer import StatevectorSimulator
            simulator = StatevectorSimulator()
            print("Using BasicAer StatevectorSimulator")
            
except ImportError as e:
    print(f"Import error: {e}")
    print("This script requires qiskit. Try running: pip install qiskit qiskit-aer matplotlib scipy")
    import sys
    sys.exit(1)

class COBYLAOptimizer:
    """Custom COBYLA optimizer with visualization capabilities."""
    
    def __init__(self, cost_function, initial_params, constraints=None):
        """
        Initialize COBYLA optimizer.
        
        Args:
            cost_function: Function to minimize
            initial_params: Initial parameter values
            constraints: Optional constraints for optimization
        """
        self.cost_function = cost_function
        self.initial_params = np.array(initial_params)
        self.constraints = constraints
        self.history = {
            'params': [],
            'costs': [],
            'gradients': []
        }
        self.iteration = 0
        
    def callback(self, xk):
        """Callback function to track optimization progress."""
        cost = self.cost_function(xk)
        self.history['params'].append(xk.copy())
        self.history['costs'].append(cost)
        self.iteration += 1
        
        if self.iteration % 10 == 0:
            print(f"Iteration {self.iteration}: Cost = {cost:.6f}")
    
    def optimize(self, maxiter=100, rhobeg=1.0, rhoend=1e-4):
        """
        Run COBYLA optimization.
        
        Args:
            maxiter: Maximum number of iterations
            rhobeg: Initial trust region radius
            rhoend: Final trust region radius
            
        Returns:
            OptimizeResult object from scipy
        """
        print("Starting COBYLA optimization...")
        
        # Wrapper to track function calls
        def wrapped_cost(params):
            return self.cost_function(params)
        
        # Convert equality constraints to inequality constraints for COBYLA
        cobyla_constraints = []
        if self.constraints:
            for constraint in self.constraints:
                if constraint['type'] == 'eq':
                    # Convert equality constraint to two inequality constraints
                    # f(x) = 0 becomes f(x) >= -eps and f(x) <= eps
                    # Which translates to: f(x) + eps >= 0 and -f(x) + eps >= 0
                    eps = 1e-6
                    cobyla_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, f=constraint['fun']: f(x) + eps
                    })
                    cobyla_constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, f=constraint['fun']: -f(x) + eps
                    })
                else:
                    cobyla_constraints.append(constraint)
        
        # Run optimization
        result = minimize(
            wrapped_cost,
            self.initial_params,
            method='COBYLA',
            callback=self.callback,
            options={
                'maxiter': maxiter,
                'rhobeg': rhobeg,
                'rhoend': rhoend,
                'disp': True
            },
            constraints=cobyla_constraints if cobyla_constraints else None
        )
        
        print(f"\nOptimization completed in {self.iteration} iterations")
        print(f"Final cost: {result.fun:.6f}")
        print(f"Optimal parameters: {result.x}")
        
        return result
    
    def plot_convergence(self, filename='cobyla_convergence.png'):
        """Plot the convergence of the optimization."""
        plt.figure(figsize=(10, 6))
        
        # Plot cost function evolution
        plt.subplot(2, 1, 1)
        plt.plot(self.history['costs'], 'b-', linewidth=2)
        plt.ylabel('Cost Function', fontsize=12)
        plt.title('COBYLA Optimization Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Plot parameter evolution
        plt.subplot(2, 1, 2)
        params_array = np.array(self.history['params'])
        for i in range(params_array.shape[1]):
            plt.plot(params_array[:, i], label=f'θ{i}', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Parameter Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Convergence plot saved as '{filename}'")
        plt.show()
    
    def plot_landscape_2d(self, param_indices=(0, 1), resolution=50, 
                          filename='cobyla_landscape.png'):
        """
        Plot 2D optimization landscape for two parameters.
        
        Args:
            param_indices: Tuple of parameter indices to plot
            resolution: Grid resolution
            filename: Output filename
        """
        if len(self.initial_params) < 2:
            print("Need at least 2 parameters for 2D landscape plot")
            return
        
        # Create grid
        optimal_params = self.history['params'][-1]
        idx1, idx2 = param_indices
        
        p1_range = np.linspace(optimal_params[idx1] - 2, optimal_params[idx1] + 2, resolution)
        p2_range = np.linspace(optimal_params[idx2] - 2, optimal_params[idx2] + 2, resolution)
        P1, P2 = np.meshgrid(p1_range, p2_range)
        
        # Calculate cost landscape
        Z = np.zeros_like(P1)
        test_params = optimal_params.copy()
        
        for i in range(resolution):
            for j in range(resolution):
                test_params[idx1] = P1[i, j]
                test_params[idx2] = P2[i, j]
                Z[i, j] = self.cost_function(test_params)
        
        # Plot
        fig = plt.figure(figsize=(12, 10))
        
        # 3D surface plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf = ax1.plot_surface(P1, P2, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel(f'θ{idx1}')
        ax1.set_ylabel(f'θ{idx2}')
        ax1.set_zlabel('Cost')
        ax1.set_title('Cost Function Landscape')
        
        # Plot optimization path
        path_params = np.array(self.history['params'])
        path_costs = np.array(self.history['costs'])
        ax1.plot(path_params[:, idx1], path_params[:, idx2], path_costs, 
                'r.-', linewidth=2, markersize=8)
        
        # Contour plot
        ax2 = fig.add_subplot(2, 2, 2)
        contour = ax2.contour(P1, P2, Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.plot(path_params[:, idx1], path_params[:, idx2], 'r.-', 
                linewidth=2, markersize=8)
        ax2.plot(path_params[0, idx1], path_params[0, idx2], 'go', 
                markersize=12, label='Start')
        ax2.plot(path_params[-1, idx1], path_params[-1, idx2], 'r*', 
                markersize=15, label='End')
        ax2.set_xlabel(f'θ{idx1}')
        ax2.set_ylabel(f'θ{idx2}')
        ax2.set_title('Optimization Path')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cost evolution
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(self.history['costs'], 'b-', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Cost')
        ax3.set_title('Cost Function Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Parameter evolution
        ax4 = fig.add_subplot(2, 2, 4)
        for i in range(path_params.shape[1]):
            ax4.plot(path_params[:, i], label=f'θ{i}', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Parameter Value')
        ax4.set_title('Parameter Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Landscape plot saved as '{filename}'")
        plt.show()

def create_test_circuit(params):
    """
    Create a test quantum circuit for optimization.
    
    Args:
        params: Parameter values [theta1, theta2, theta3]
        
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(2)
    
    # Layer 1
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    
    # Entanglement
    qc.cx(0, 1)
    
    # Layer 2
    qc.ry(params[2], 0)
    
    return qc

def quadratic_cost_function(params):
    """
    Simple quadratic cost function for testing.
    Minimum at params = [1, -0.5, 2]
    """
    target = np.array([1.0, -0.5, 2.0])
    diff = params - target
    return np.dot(diff, diff) + 0.1 * np.sin(5 * params[0]) * np.cos(3 * params[1])

def quantum_cost_function(params):
    """
    Quantum circuit-based cost function.
    Measures deviation from target state |01⟩.
    """
    qc = create_test_circuit(params)
    
    # Get statevector
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    
    # Target is |01⟩ state (index 1 in computational basis)
    target_prob = probs[1]  # Probability of measuring |01⟩
    
    # Cost is 1 - probability of target state
    return 1 - target_prob

def demonstrate_cobyla():
    """Demonstrate COBYLA optimization on different problems."""
    
    print("=" * 60)
    print("COBYLA Optimization Demonstration")
    print("=" * 60)
    
    # Test 1: Classical quadratic function
    print("\n1. Optimizing classical quadratic function...")
    print("   Target minimum at θ = [1.0, -0.5, 2.0]")
    
    initial_params = np.random.rand(3) * 4 - 2  # Random start in [-2, 2]
    optimizer1 = COBYLAOptimizer(quadratic_cost_function, initial_params)
    result1 = optimizer1.optimize(maxiter=50)
    
    optimizer1.plot_convergence('cobyla_quadratic_convergence.png')
    optimizer1.plot_landscape_2d(param_indices=(0, 1), 
                                filename='cobyla_quadratic_landscape.png')
    
    # Test 2: Quantum circuit optimization
    print("\n" + "=" * 60)
    print("2. Optimizing quantum circuit to prepare |01⟩ state...")
    
    initial_params = np.random.rand(3) * 2 * np.pi
    optimizer2 = COBYLAOptimizer(quantum_cost_function, initial_params)
    result2 = optimizer2.optimize(maxiter=100)
    
    # Verify result
    qc_final = create_test_circuit(result2.x)
    sv_final = Statevector.from_instruction(qc_final)
    probs_final = sv_final.probabilities()
    
    print(f"\nFinal state probabilities:")
    print(f"  |00⟩: {probs_final[0]:.4f}")
    print(f"  |01⟩: {probs_final[1]:.4f} (target)")
    print(f"  |10⟩: {probs_final[2]:.4f}")
    print(f"  |11⟩: {probs_final[3]:.4f}")
    
    optimizer2.plot_convergence('cobyla_quantum_convergence.png')
    optimizer2.plot_landscape_2d(param_indices=(0, 1), 
                                filename='cobyla_quantum_landscape.png')
    
    # Test 3: Constrained optimization
    print("\n" + "=" * 60)
    print("3. Constrained optimization (parameters sum to π)...")
    print("   Note: COBYLA converts equality constraints to inequality constraints")
    
    # Constraint: sum of parameters equals π
    # COBYLA will convert this to two inequality constraints internally
    constraint = {
        'type': 'eq',
        'fun': lambda x: np.sum(x) - np.pi
    }
    
    initial_params = np.random.rand(3) * 2
    optimizer3 = COBYLAOptimizer(quantum_cost_function, initial_params, 
                                constraints=[constraint])
    result3 = optimizer3.optimize(maxiter=100)
    
    print(f"\nSum of optimal parameters: {np.sum(result3.x):.6f} (should be ≈ π = {np.pi:.6f})")
    print(f"Constraint satisfaction: |sum - π| = {abs(np.sum(result3.x) - np.pi):.6e}")
    
    optimizer3.plot_convergence('cobyla_constrained_convergence.png')
    
    # Test 4: Inequality constrained optimization
    print("\n" + "=" * 60)
    print("4. Inequality constrained optimization...")
    print("   Constraint: All parameters must be positive and sum < 2π")
    
    # Multiple inequality constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},  # x[0] >= 0
        {'type': 'ineq', 'fun': lambda x: x[1]},  # x[1] >= 0
        {'type': 'ineq', 'fun': lambda x: x[2]},  # x[2] >= 0
        {'type': 'ineq', 'fun': lambda x: 2*np.pi - np.sum(x)}  # sum <= 2π
    ]
    
    initial_params = np.random.rand(3) * np.pi
    optimizer4 = COBYLAOptimizer(quantum_cost_function, initial_params, 
                                constraints=constraints)
    result4 = optimizer4.optimize(maxiter=100)
    
    print(f"\nOptimal parameters: {result4.x}")
    print(f"All positive: {all(x >= 0 for x in result4.x)}")
    print(f"Sum: {np.sum(result4.x):.6f} (should be < 2π = {2*np.pi:.6f})")
    
    optimizer4.plot_convergence('cobyla_inequality_convergence.png')

if __name__ == "__main__":
    demonstrate_cobyla()
    
    print("\n" + "=" * 60)
    print("COBYLA optimization demonstration completed!")
    print("Check the generated plots for visualization of the optimization process.")
