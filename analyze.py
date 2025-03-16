# Analysis and visualization for quantum-enhanced model

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import *

def visualize_attention(model, tokenizer, sentence, save_path=None):
    """Visualize attention patterns for a sentence"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get attention weights - this depends on the specific model structure
    try:
        # For DistilBERT
        attention = model.distilbert.transformer.layer[0].attention.dropout(
            model.distilbert.transformer.layer[0].attention.q_lin(
                model.distilbert.transformer.layer[0].attention.get_scores(inputs.input_ids, inputs.attention_mask)
            )
        )
    except:
        print("Cannot extract attention weights directly for visualization")
        return
    
    # Plot attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention[0].cpu().numpy(), cmap="viridis")
    plt.title("Attention Heatmap")
    plt.colorbar()
    
    # Get tokens for labels
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def compare_with_baseline(quantum_model_path, baseline_model_path=None):
    """Compare quantum-enhanced model with baseline model"""
    # Load quantum-enhanced model results
    with open(f"{quantum_model_path}/results.json", "r") as f:
        quantum_results = json.load(f)
    
    # Load or train baseline model if needed
    if baseline_model_path and os.path.exists(f"{baseline_model_path}/results.json"):
        with open(f"{baseline_model_path}/results.json", "r") as f:
            baseline_results = json.load(f)
    else:
        print("No baseline results found. Baseline comparison skipped.")
        return quantum_results, None
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(quantum_results["train_losses"], 'b-', label='Quantum-Enhanced')
    plt.plot(baseline_results["train_losses"], 'r--', label='Classical')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy comparison
    plt.subplot(1, 2, 2)
    plt.plot(quantum_results["eval_accuracies"], 'b-', label='Quantum-Enhanced')
    plt.plot(baseline_results["eval_accuracies"], 'r--', label='Classical')
    plt.title('Evaluation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{quantum_model_path}/comparison.png")
    plt.show()
    
    # Print final metrics
    print("Final Results:")
    print(f"Quantum-Enhanced - Final Loss: {quantum_results['train_losses'][-1]:.4f}, Final Accuracy: {quantum_results['eval_accuracies'][-1]:.4f}")
    print(f"Classical Baseline - Final Loss: {baseline_results['train_losses'][-1]:.4f}, Final Accuracy: {baseline_results['eval_accuracies'][-1]:.4f}")
    
    return quantum_results, baseline_results

def analyze_model(model_path):
    """Run analysis on a trained model"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Example sentences for attention visualization
    positive_sentence = "This movie was absolutely wonderful and I enjoyed every minute of it."
    negative_sentence = "This film was a complete waste of time and money."
    
    try:
        # Try to visualize attention patterns
        visualize_attention(model, tokenizer, positive_sentence, save_path=f"{model_path}/attention_positive.png")
        visualize_attention(model, tokenizer, negative_sentence, save_path=f"{model_path}/attention_negative.png")
    except Exception as e:
        print(f"Attention visualization failed: {e}")
    
    return model, tokenizer

if __name__ == "__main__":
    # Find the latest run
    if os.path.exists(RESULTS_DIR):
        result_dirs = [d for d in os.listdir(RESULTS_DIR) if d.startswith("run_")]
        if result_dirs:
            latest_run = sorted(result_dirs)[-1]
            model_path = f"{RESULTS_DIR}/{latest_run}"
            print(f"Analyzing run: {model_path}")
            
            # Load results
            with open(f"{model_path}/results.json", "r") as f:
                results = json.load(f)
            
            # Print key metrics
            print("\n=== TRAINING RESULTS ===")
            print(f"Final accuracy: {results['eval_accuracies'][-1]:.4f}")
            print(f"Training loss: {results['train_losses'][-1]:.4f}")
            print(f"Backend used: {results['config']['backend']}")
            print("\nFull accuracy history:")
            for i, acc in enumerate(results['eval_accuracies']):
                print(f"Epoch {i+1}: {acc:.4f}")
    else:
        print("No results found")