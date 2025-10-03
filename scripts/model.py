# File: model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # d_model: dimension of the embedding vector
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention block
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # Feed-forward block
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        return src

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(encoder_layers)
        self.d_model = d_model

    def forward(self, src, src_pad_mask):
        # src shape: [batch_size, src_len]
        # src_pad_mask shape: [batch_size, src_len]
        
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1) # Transpose for PE
        
        # Pass through encoder layers
        for layer in self.layers:
            src = layer(src, src_mask=src_pad_mask)
            
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # This is the cross-attention layer
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # Self-attention block (for target sentence)
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        # Cross-attention block (target attends to encoder memory)
        attn_output, _ = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)

        # Feed-forward block
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)
        return tgt

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layers = [TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(decoder_layers)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # Embedding and positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt.transpose(0, 1)).transpose(0, 1)

        # Pass through decoder layers
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
            
        return tgt

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward):
        super(Seq2SeqTransformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward)
        self.decoder = Decoder(tgt_vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_lookahead_mask):
        memory = self.encoder(src, src_pad_mask)
        output = self.decoder(tgt, memory, tgt_lookahead_mask, src_pad_mask)
        logits = self.generator(output)
        return logits