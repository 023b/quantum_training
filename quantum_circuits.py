# Quantum circuits for parameter updates

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler  # Local sampler
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as IBMSampler  # IBM sampler
import numpy as np
import os
from config import *

def initialize_quantum_backend():
    """Initialize the IBM Quantum backend or fall back to local simulator"""
    try:
        # Try to initialize the service with existing credentials
        print("Attempting to connect to IBM Quantum...")
        service = QiskitRuntimeService()
        
        # List available backends
        available_backends = service.backends()
        backend_names = [b.name for b in available_backends]
        print("Available backends:", backend_names)
        
        # Try to select backend from config or use first available one
        if BACKEND_NAME in backend_names:
            backend = service.backend(BACKEND_NAME)
        else:
            # If config backend not available, use the first one
            backend = service.backend(backend_names[0])
        print(f"Using IBM Quantum backend: {backend.name}")
        
        return service, backend
        
    except Exception as e:
        print(f"Error connecting to IBM Quantum: {e}")
        print("Falling back to local simulator...")
        
        # Try different approaches to get a simulator
        try:
            # Try Aer simulator (newer versions)
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
            print(f"Using Aer simulator: {backend.name}")
        except ImportError:
            try:
                # Try qiskit.Aer (medium-old versions)
                from qiskit import Aer
                backend = Aer.get_backend('qasm_simulator')
                print(f"Using Aer simulator: {backend.name}")
            except ImportError:
                # As a last resort, use Sampler without a specific backend
                from qiskit.primitives import Sampler
                print("Using primitive Sampler as simulator")
                backend = None  # We'll use Sampler directly in quantum_parameter_update
        
        return None, backend

def create_quantum_circuit(num_qubits=NUM_QUBITS):
    """Create a parameterized quantum circuit for parameter updates"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Parameters for rotation gates
    params = [Parameter(f"θ{i}") for i in range(num_qubits)]
    
    # Apply Hadamard gates to create superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Apply parameterized rotation gates
    for i in range(num_qubits):
        qc.ry(params[i], i)
    
    # Add entanglement
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    
    # Add second layer of rotations
    for i in range(num_qubits):
        qc.ry(params[i]/2, i)
    
    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))
    
    return qc, params

def quantum_parameter_update(gradient_values, backend=None, learning_rate=QUANTUM_LEARNING_RATE, shots=SHOTS):
    """Use quantum circuit to enhance parameter updates"""
    num_params = min(NUM_QUBITS, len(gradient_values))  # Limit to NUM_QUBITS
    
    # Create circuit and get parameters
    qc, params = create_quantum_circuit(num_params)
    
    # Normalize gradient values to suitable range for rotation gates
    norm_factor = max(abs(np.max(gradient_values)), abs(np.min(gradient_values)))
    if norm_factor == 0:
        norm_factor = 1.0
    normalized_gradients = [g/norm_factor * np.pi for g in gradient_values[:num_params]]
    
    # Create parameter dictionary
    param_dict = {params[i]: normalized_gradients[i] for i in range(num_params)}
    
    # Bind parameters to circuit - use assign_parameters for compatibility
    try:
        bound_qc = qc.bind_parameters(param_dict)
    except AttributeError:
        bound_qc = qc.assign_parameters(param_dict)
    
    # Always use local Sampler for simplicity and compatibility
    from qiskit.primitives import Sampler
    sampler = Sampler()
    job = sampler.run(bound_qc, shots=shots)
    
    result = job.result()
    counts = result.quasi_dists[0]
    
    # Process measurement results to create update direction
    updates = np.zeros(len(gradient_values))
    for bitstring, probability in counts.items():
        # Convert bitstring to parameter updates
        if isinstance(bitstring, int):
            bit_array = [int(b) for b in format(bitstring, f'0{num_params}b')]
        else:
            bit_array = [int(b) for b in format(int(bitstring), f'0{num_params}b')]
        
        # Use measured bits to modify gradient direction
        for i in range(num_params):
            if i < len(bit_array):
                factor = 1 if bit_array[i] == 1 else -0.5
                updates[i] += factor * probability
    
    # Scale updates by learning rate and extend to original size
    updates = updates * learning_rate
    if len(updates) < len(gradient_values):
        updates = np.pad(updates, (0, len(gradient_values) - len(updates)))
    
    return updates