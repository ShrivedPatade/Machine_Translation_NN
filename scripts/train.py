# File: train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
import os
import time
import math

# Import our custom modules
from dataset import get_dataloader
from model import Seq2SeqTransformer

# =============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =============================================================================
DATA_PATH = '../parallel-corpus/data/final_data/'
TOKENIZER_PATH = 'english_sanskrit_bpe_tokenizer.json'
MODEL_SAVE_PATH = '../models/best_model.pth'

# Model Hyperparameters
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
D_MODEL = 256
NHEAD = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Training Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
ACCUMULATION_STEPS = 4  # For gradient accumulation

# =============================================================================
# 2. SETUP (DEVICE, TOKENIZER, DATALOADERS)
# =============================================================================
# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Tokenizer and DataLoaders ---
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
SRC_VOCAB_SIZE = tokenizer.get_vocab_size()
TGT_VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_IDX = tokenizer.token_to_id("[PAD]")

train_loader = get_dataloader(DATA_PATH, TOKENIZER_PATH, 'train', BATCH_SIZE)
val_loader = get_dataloader(DATA_PATH, TOKENIZER_PATH, 'dev', BATCH_SIZE)

# =============================================================================
# 3. HELPER FUNCTION FOR MASK CREATION
# =============================================================================
def create_masks(src, tgt, device):
    # Source padding mask: (batch_size, src_len)
    # True where src is PAD_IDX, False otherwise.
    src_pad_mask = (src == PAD_IDX).to(device)

    # Target padding mask: (batch_size, tgt_len)
    tgt_pad_mask = (tgt == PAD_IDX).to(device)

    # Target look-ahead mask (subsequent mask)
    # This prevents the decoder from "cheating" by looking at future tokens.
    tgt_len = tgt.shape[1]
    tgt_lookahead_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) == 1).transpose(0, 1)
    tgt_lookahead_mask = tgt_lookahead_mask.float().masked_fill(tgt_lookahead_mask == 0, float('-inf')).masked_fill(tgt_lookahead_mask == 1, float(0.0))
    
    return src_pad_mask, tgt_pad_mask, tgt_lookahead_mask

# =============================================================================
# 4. TRAINING AND EVALUATION LOGIC
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    start_time = time.time()

    # Zero gradients at the start of the epoch
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_pad_mask, tgt_pad_mask, tgt_lookahead_mask = create_masks(src, tgt_input, device)

        # Forward pass
        logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask, tgt_lookahead_mask)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        # Normalize loss to account for accumulation
        loss = loss / ACCUMULATION_STEPS

        # Backward pass (accumulates gradients)
        loss.backward()

        # --- Gradient Accumulation Step ---
        # Update weights only every ACCUMULATION_STEPS
        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad() # Reset gradients for the next accumulation cycle

        total_loss += loss.item() * ACCUMULATION_STEPS # De-normalize loss for logging

        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {i}/{len(dataloader)} | Loss: {(loss.item() * ACCUMULATION_STEPS):.4f} | Time: {elapsed:.2f}s")
            start_time = time.time()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_pad_mask, tgt_pad_mask, tgt_lookahead_mask = create_masks(src, tgt_input, device)

            logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask, tgt_lookahead_mask)
            
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

# =============================================================================
# 5. MAIN ORCHESTRATION BLOCK
# =============================================================================
if __name__ == "__main__":
    # --- Model Initialization ---
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, D_MODEL, NHEAD,
                               SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, DIM_FEEDFORWARD).to(device)
    
    # Initialize weights with Xavier uniform distribution for better training stability
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # --- Loss and Optimizer ---
    # We use CrossEntropyLoss, ignoring the PAD token index
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Main Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        
        print(f"--- Starting Epoch {epoch}/{NUM_EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        
        epoch_time = time.time() - start_time
        
        print("-" * 50)
        print(f"| End of Epoch {epoch} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} |")
        print("-" * 50)

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation loss improved. Saving model to {MODEL_SAVE_PATH}")