# Model and dataset utilities

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import os
from config import *

def load_tokenizer_and_model():
    """Load the tokenizer and model"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Identify key parameters to enhance with quantum computing
    attention_weights = model.distilbert.transformer.layer[0].attention.out_lin.weight
    print(f"Attention weight shape: {attention_weights.shape}")
    
    return tokenizer, model

def prepare_datasets(tokenizer):
    """Prepare the dataset for training and evaluation"""
    # Load dataset
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    
    # Select small subsets for experiments
    train_dataset = dataset["train"].select(range(TRAIN_SIZE))
    eval_dataset = dataset["validation"].select(range(EVAL_SIZE))
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"], 
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # Rename 'label' to 'labels' to match model's expected input
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")
    
    # Set format for PyTorch
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create DataLoaders
    train_dataloader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(tokenized_eval, batch_size=BATCH_SIZE)
    
    return train_dataloader, eval_dataloader, train_dataset, eval_dataset

def evaluate_model(model, eval_dataloader, device):
    """Evaluate the model on the evaluation dataset"""
    model.eval()
    correct = 0
    total = 0
    
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != 'idx'}
        with torch.no_grad():
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    accuracy = correct / total
    return accuracy

def save_model(model, tokenizer, model_name, save_dir=SAVE_DIR):
    """Save the model and tokenizer"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, model_name)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")