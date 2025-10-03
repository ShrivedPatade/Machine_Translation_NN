# File: train_tokenizer.py
# Purpose: Trains a custom BPE tokenizer from the preprocessed training files.

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- Configuration ---
# Path to the directory containing our final train/dev/test files
FINAL_DATA_PATH = '../parallel-corpus/data/final_data/'
# The vocabulary size for our tokenizer
VOCAB_SIZE = 32000
# The final name for our saved tokenizer file
TOKENIZER_SAVE_PATH = "english_sanskrit_bpe_tokenizer.json"

# --- 1. Gather the training files ---
# We ONLY train the tokenizer on the training set to avoid data leakage
files_to_train_on = [
    os.path.join(FINAL_DATA_PATH, 'train.en'),
    os.path.join(FINAL_DATA_PATH, 'train.sa')
]

# Check if the training files actually exist
for f in files_to_train_on:
    if not os.path.exists(f):
        print(f"FATAL ERROR: Training file not found at {f}")
        print("Please ensure the preprocess.py script has been run successfully.")
        exit()

# --- 2. Define and Train the Tokenizer ---
# Initialize a new tokenizer with a BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Set the pre-tokenizer (this splits by whitespace before BPE is applied)
tokenizer.pre_tokenizer = Whitespace()

# Configure the trainer
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE, 
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"] # Unknown, Padding, Start, End
)

# Train the tokenizer on our files
print(f"Training tokenizer from files: {files_to_train_on}...")
tokenizer.train(files=files_to_train_on, trainer=trainer)

# --- 3. Save the Trained Tokenizer ---
tokenizer.save(TOKENIZER_SAVE_PATH)

print("\n----------------------------------------------------")
print("Tokenizer training complete!")
print(f"Tokenizer with vocab size {tokenizer.get_vocab_size()} saved to: {TOKENIZER_SAVE_PATH}")
print("----------------------------------------------------")