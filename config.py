# Configuration settings for the quantum-enhanced model fine-tuning

# IBM Quantum settings
IBM_TOKEN = "YOUR_IBM_QUANTUM_TOKEN"  # Not needed if already set up
BACKEND_NAME = "ibm_brisbane"  # Using one of your available backends

# Model settings
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 128

# Dataset settings
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"
TRAIN_SIZE = 100  # Small subset for experiments
EVAL_SIZE = 50

# Training settings
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
QUANTUM_LEARNING_RATE = 0.01
NUM_QUBITS = 5
SHOTS = 1000

# Paths
SAVE_DIR = "saved_models"
RESULTS_DIR = "results"