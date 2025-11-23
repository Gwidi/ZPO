import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()

        # Transform input into queries, keys, and values
        # Step 1: Apply self.qkv linear layer to get combined Q, K, V
        # Step 2: Reshape to separate Q, K, V and split into multiple heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Compute attention scores by calculating scaled dot-product attention
        # Step 1: Compute attention scores: Q @ K^T / sqrt(head_dim)
        # Step 2: Apply softmax to get attention weights
        # Step 3: Apply attention weights to values: attention_weights @ V
        # Combine multi-head outputs
        # Step 1: Reshape to merge all heads
        # Step 2: Apply final linear projection
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', qkv[:,:,0], qkv[:,:,1]) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, qkv[:,:,2])
        attention_output = attention_output.reshape(batch_size, seq_len, self.embed_dim)     

        output = self.fc_out(attention_output)

        return output


if __name__ == "__main__":
    embed_dim = 768
    patches = torch.randn(32, 196, embed_dim)  # (batch_size, num_patches, embed_dim)
    attn = MultiHeadSelfAttention(embed_dim, 12)
    out = attn(patches)
    assert out.shape == patches.shape, "Attention output shape incorrect"