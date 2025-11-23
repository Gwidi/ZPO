import torch
import torch.nn as nn

from multi_head_attention import MultiHeadSelfAttention
from patch_embedding import PatchEmbedding


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder using Pre-Norm strategy.

    Normalization strategy:
    - LayerNorm is applied before both the self-attention and the MLP sub-layers (Pre-Norm).
    - Each sub-layer (self-attention and MLP) is followed by a residual connection.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        # Pre-Norm: Apply LayerNorm before self-attention, then add residual connection
        x = x + self.attn(self.norm1(x))

        # Pre-Norm: Apply LayerNorm before MLP, then add residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stacks multiple TransformerEncoderLayer instances."""
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model for image classification."""
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            num_classes: int = 37,
            embed_dim: int = 768,
            num_heads: int = 12,
            num_layers: int = 12,
            mlp_ratio: float = 4.,
            dropout: float = 0.1,
        ) -> None:
        """
        Initialize the Vision Transformer model

        Parameters
        ----------
        img_size : int, optional
            Size of the input image (assumed square), by default 224
        patch_size : int, optional
            Size of each patch (assumed square), by default 16
        in_channels : int, optional
            Number of input channels in the image, by default 3
        num_classes : int, optional
            Number of output classes for classification, by default 37
        embed_dim : int, optional
            Dimensionality of the token embeddings, by default 768
        num_heads : int, optional
            Number of attention heads in the transformer encoder, by default 12
        num_layers : int, optional
            Number of transformer encoder layers, by default 12
        mlp_ratio : float, optional
            Ratio of MLP hidden dimension to embedding dimension, by default 4.
        dropout : float, optional
            Dropout probability, by default 0.1
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches

        # Add a learnable class token
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))

        # Positional embeddings for patches and the class token
        # Add 1 for the class token
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Create transformer encoder using custom implementation
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize positional embedding and class token
        nn.init.trunc_normal_(self.positional_embedding, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Vision Transformer."""
        # x: (batch_size, in_channels, img_size, img_size)

        # Apply patch embedding
        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)

        # Prepend the class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.positional_embedding[:, :x.size(1)]

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x) # (batch_size, num_patches + 1, embed_dim)

        # Extract the output corresponding to the class token
        # In ViT, only the first token (class token) is used for classification.
        cls_token_output = x[:, 0] # (batch_size, embed_dim)

        # Apply normalization and classification head
        output = self.head(self.norm(cls_token_output)) # (batch_size, num_classes)

        return output