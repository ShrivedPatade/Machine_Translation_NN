# File: train.py (Upgraded with Checkpointing)

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
CHECKPOINT_PATH = '../models/training_checkpoint.pth' # For resuming training
BEST_MODEL_SAVE_PATH = '../models/best_model.pth'      # For the final inference model

# Model Hyperparameters 
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
D_MODEL = 256
NHEAD = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Training Hyperparameters 
BATCH_SIZE = 4 
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

# =============================================================================
# 2. SETUP (DEVICE, TOKENIZER, ETC.)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
SRC_VOCAB_SIZE = tokenizer.get_vocab_size()
TGT_VOCAB_SIZE = tokenizer.get_vocab_size()
PAD_IDX = tokenizer.token_to_id("[PAD]")

# =============================================================================
# (Helper functions: create_masks, train_epoch, evaluate - are the same as before)
# ... Copy your create_masks, train_epoch, and evaluate functions here ...
# For clarity, I'm including them again below.
# =============================================================================

def create_masks(src, tgt, device):
    src_pad_mask = (src == PAD_IDX).to(device)
    tgt_pad_mask = (tgt == PAD_IDX).to(device)
    tgt_len = tgt.shape[1]
    tgt_lookahead_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) == 1).transpose(0, 1)
    tgt_lookahead_mask = tgt_lookahead_mask.float().masked_fill(tgt_lookahead_mask == 0, float('-inf')).masked_fill(tgt_lookahead_mask == 1, float(0.0))
    return src_pad_mask, tgt_pad_mask, tgt_lookahead_mask

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    start_time = time.time()
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_pad_mask, tgt_pad_mask, tgt_lookahead_mask = create_masks(src, tgt_input, device)
        logits = model(src, tgt_input, src_pad_mask, tgt_pad_mask, tgt_lookahead_mask)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * ACCUMULATION_STEPS
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
# 5. MAIN ORCHESTRATION BLOCK (WITH CHECKPOINTING)
# =============================================================================
if __name__ == "__main__":
    train_loader = get_dataloader(DATA_PATH, TOKENIZER_PATH, 'train', BATCH_SIZE)
    val_loader = get_dataloader(DATA_PATH, TOKENIZER_PATH, 'dev', BATCH_SIZE)

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, D_MODEL, NHEAD,
                               SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, DIM_FEEDFORWARD).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # --- NEW: LOGIC FOR LOADING CHECKPOINT ---
    start_epoch = 1
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        # NOTE: Since your first save was only the model's state_dict, we handle that case.
        # Future saves will be the full dictionary.
        try:
            checkpoint = torch.load(CHECKPOINT_PATH)
            # Check if it's our new dictionary format or the old model-only format
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['best_val_loss']
                print(f"Checkpoint found. Resuming training from Epoch {start_epoch}.")
            else: # It's the old format (just model weights)
                model.load_state_dict(checkpoint)
                print(f"Found old model file. Loaded model weights. Starting from Epoch {start_epoch}.")
                print("Optimizer state not found, starting optimizer from scratch.")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")
        # Initialize weights only if not loading from a checkpoint
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    # --- Main Training Loop ---
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        start_time = time.time()
        
        print(f"--- Starting Epoch {epoch}/{NUM_EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        
        epoch_time = time.time() - start_time
        
        print("-" * 50)
        print(f"| End of Epoch {epoch} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} |")
        print("-" * 50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # --- NEW: SAVE A COMPREHENSIVE CHECKPOINT ---
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHECKPOINT_PATH)
            
            # We can also save a simple model file for easy inference later
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"Validation loss improved. Saving checkpoint to {CHECKPOINT_PATH}")