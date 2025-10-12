import math
import torch
import torch.nn as nn

class SampleRateEmbedding(nn.Module):
    """
    Creates fixed sinusoidal embeddings for a discrete set of sample rates.

    This module adapts the sinusoidal position embedding technique from the
    "Attention Is All You Need" paper to create a unique, high-dimensional,
    and non-trainable vector for each supported sample rate. These embeddings
    provide a rich representation of the sample rate for the conditioning mechanism.

    Args:
        embedding_dim (int): The dimensionality of the output embedding. Must be an even number.
        sample_rates (list): A list of integer sample rates to create embeddings for.
    """
    def __init__(self, embedding_dim, sample_rates):
        """Initializes the embedding table."""
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

        # Register the table as a buffer so it's part of the model state but not a trainable parameter.
        self.register_buffer('embedding_table', embedding_table)

    def forward(self, sr_values):
        """
        Looks up the embeddings for a batch of sample rate values.

        Args:
            sr_values (torch.Tensor): A tensor containing the sample rates for the batch.

        Returns:
            torch.Tensor: A tensor containing the corresponding embeddings.
        """
        indices = torch.tensor([self.sr_to_idx[sr.item()] for sr in sr_values], device=sr_values.device)

        return self.embedding_table[indices]