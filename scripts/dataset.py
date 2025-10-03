# File: dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
import os

class TranslationDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, split='train'):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        en_path = os.path.join(data_path, f"{split}.en")
        sa_path = os.path.join(data_path, f"{split}.sa")

        with open(en_path, 'r', encoding='utf-8') as f:
            self.english_sentences = [line.strip() for line in f.readlines()]
        
        with open(sa_path, 'r', encoding='utf-8') as f:
            self.sanskrit_sentences = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        src_text = self.english_sentences[idx]
        tgt_text = self.sanskrit_sentences[idx]

        # Add SOS and EOS tokens during tokenization
        src_encoded = self.tokenizer.encode(f"[SOS] {src_text} [EOS]")
        tgt_encoded = self.tokenizer.encode(f"[SOS] {tgt_text} [EOS]")

        return {
            "src_ids": torch.tensor(src_encoded.ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_encoded.ids, dtype=torch.long)
        }

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Separate source and target sentences
        srcs = [item['src_ids'] for item in batch]
        tgts = [item['tgt_ids'] for item in batch]

        # Use pad_sequence to pad sentences in the batch
        srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=self.pad_idx)
        tgts_padded = pad_sequence(tgts, batch_first=True, padding_value=self.pad_idx)

        return {"src": srcs_padded, "tgt": tgts_padded}

def get_dataloader(data_path, tokenizer_path, split, batch_size):
    # Get the ID for the padding token
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("[PAD]")

    dataset = TranslationDataset(data_path, tokenizer_path, split=split)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'), # Only shuffle the training set
        collate_fn=PadCollate(pad_idx=pad_token_id)
    )
    return loader

# --- Example of how to use it ---
if __name__ == '__main__':
    DATA_PATH = '../parallel-corpus/data/final_data/'
    TOKENIZER_PATH = 'english_sanskrit_bpe_tokenizer.json'
    BATCH_SIZE = 4 # Small batch size for testing

    train_loader = get_dataloader(DATA_PATH, TOKENIZER_PATH, 'train', BATCH_SIZE)

    # Get one batch of data
    batch = next(iter(train_loader))
    
    print("Batch of source sentences (padded):")
    print(batch['src'])
    print("Shape:", batch['src'].shape) # e.g., [4, max_src_length_in_batch]
    
    print("\nBatch of target sentences (padded):")
    print(batch['tgt'])
    print("Shape:", batch['tgt'].shape) # e.g., [4, max_tgt_length_in_batch]