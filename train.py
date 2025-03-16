# Training script for quantum-enhanced model fine-tuning

import torch
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from quantum_circuits import initialize_quantum_backend, quantum_parameter_update
from model_utils import load_tokenizer_and_model, prepare_datasets, evaluate_model, save_model
from config import *

def train_quantum_enhanced_model():
    """Train a model with quantum-enhanced parameter updates"""
    # Initialize quantum backend
    service, backend = initialize_quantum_backend()
    
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model()
    
    # Prepare datasets
    train_dataloader, eval_dataloader, _, _ = prepare_datasets(tokenizer)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Track metrics
    train_losses = []
    eval_accuracies = []
    
    # Save original parameters for comparison
    original_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"{RESULTS_DIR}/run_{timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Training loop
    print(f"Starting training on {device}...")
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'idx'}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Backward pass
            loss.backward()
            
            # Quantum-enhanced update for attention layer
            with torch.no_grad():
                # Get gradients
                attention_grads = model.distilbert.transformer.layer[0].attention.out_lin.weight.grad.flatten().cpu().numpy()
                
                # Get quantum-enhanced updates
                quantum_updates = quantum_parameter_update(attention_grads, backend)
                
                # Reshape back to original shape
                quantum_updates = quantum_updates.reshape(model.distilbert.transformer.layer[0].attention.out_lin.weight.shape)
                
                # Apply quantum updates to attention weights
                model.distilbert.transformer.layer[0].attention.out_lin.weight -= torch.tensor(quantum_updates).to(device)
            
            # Regular update for other parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Evaluation
        accuracy = evaluate_model(model, eval_dataloader, device)
        eval_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Eval Accuracy: {accuracy:.4f}")
    
    # Save results
    results = {
    "train_losses": train_losses,
    "eval_accuracies": eval_accuracies,
    "config": {
        "model_name": MODEL_NAME,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "quantum_learning_rate": QUANTUM_LEARNING_RATE,
        "num_qubits": NUM_QUBITS,
        "shots": SHOTS,
        "backend": "local_simulator" if backend is None else backend.name,
            }
        }
    
    with open(f"{result_dir}/results.json", "w") as f:
        json.dump(results, f)
    
    # Save model
    save_model(model, tokenizer, f"quantum_enhanced_{timestamp}", save_dir=result_dir)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(eval_accuracies, 'b-')
    plt.title('Evaluation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/training_curves.png")
    plt.show()
    
    return model, tokenizer, results

if __name__ == "__main__":
    train_quantum_enhanced_model()