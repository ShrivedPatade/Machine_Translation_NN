# File: translate.py (Corrected with matching hyperparameters)

import torch
from tokenizers import Tokenizer
import sys

# Import our custom model architecture
from model import Seq2SeqTransformer

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# --- Paths ---
TOKENIZER_PATH = 'english_sanskrit_bpe_tokenizer.json'
MODEL_PATH = '../models/best_model.pth' 

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Hyperparameters ---
# *** CORRECTED: These now EXACTLY MATCH your successful training parameters ***
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
D_MODEL = 256
NHEAD = 4
DIM_FEEDFORWARD = 1024

# =============================================================================
# 2. TRANSLATION FUNCTION (Unchanged)
# =============================================================================
def translate(model, sentence, tokenizer):
    model.eval()
    sos_idx = tokenizer.token_to_id("[SOS]")
    eos_idx = tokenizer.token_to_id("[EOS]")
    pad_idx = tokenizer.token_to_id("[PAD]")

    src_tokens = [sos_idx] + tokenizer.encode(sentence.lower()).ids + [eos_idx]
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    src_pad_mask = (src_tensor == pad_idx).to(device)

    with torch.no_grad():
        memory = model.encoder(src_tensor, src_pad_mask)

    tgt_tokens = [sos_idx]
    
    for i in range(100):
        tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
        tgt_len = tgt_tensor.shape[1]
        tgt_lookahead_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) == 1).transpose(0, 1)
        tgt_lookahead_mask = tgt_lookahead_mask.float().masked_fill(tgt_lookahead_mask == 0, float('-inf')).masked_fill(tgt_lookahead_mask == 1, float(0.0))
        
        with torch.no_grad():
            output = model.decoder(tgt_tensor, memory, tgt_lookahead_mask, src_pad_mask)
            pred_token = model.generator(output[:, -1]).argmax(dim=-1).item()

        tgt_tokens.append(pred_token)

        if pred_token == eos_idx:
            break

    return tokenizer.decode(tgt_tokens[1:])

# =============================================================================
# 3. MAIN EXECUTION BLOCK (Updated with correct model init)
# =============================================================================
if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    SRC_VOCAB_SIZE = tokenizer.get_vocab_size()
    TGT_VOCAB_SIZE = tokenizer.get_vocab_size()

    # --- Initialize Model with the CORRECT architecture ---
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, D_MODEL, NHEAD,
                               SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, DIM_FEEDFORWARD).to(device)
    
    print(f"Loading model weights from {MODEL_PATH}")
    # --- Load Trained Weights ---
    # Added weights_only=True to address the FutureWarning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    if len(sys.argv) > 1:
        source_sentence = sys.argv[1]
    else:
        source_sentence = "A boy is reading a book."

    print("\n----------------------------------------------------")
    print(f"English Source:  '{source_sentence}'")
    
    translation = translate(model, source_sentence, tokenizer)
    
    print(f"Sanskrit Target: '{translation}'")
    print("----------------------------------------------------")