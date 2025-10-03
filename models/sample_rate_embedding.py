import math
import torch
import torch.nn as nn

class SampleRateEmbedding(nn.Module):
    def __init__(self, embedding_dim, sample_rates):
        super().__init__()
        if embedding_dim % 2 != 0: raise ValueError("Embedding dim must be even.")
        self.embedding_dim = embedding_dim
        self.sr_to_idx = {sr: i for i, sr in enumerate(sample_rates)}
        
        num_embeddings = len(sample_rates)
        position = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        embedding_table = torch.zeros(num_embeddings, embedding_dim)
        embedding_table[:, 0::2] = torch.sin(position * div_term)
        embedding_table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('embedding_table', embedding_table)

    def forward(self, sr_values):
        indices = torch.tensor([self.sr_to_idx[sr.item()] for sr in sr_values], device=sr_values.device)

        return self.embedding_table[indices]