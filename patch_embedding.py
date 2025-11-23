import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Splits images into patches and applies linear projection."""
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        self.num_patches = (img_size // patch_size) ** 2

        # Create a Conv2d projection layer that will split the image into patches and project them.
        # Layer configuration:
        #   - Takes in_channels as input
        #   - Outputs embed_dim channels
        #   - Uses kernel_size and stride equal to patch_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        # input x: (batch_size, in_channels, img_size, img_size)
        # Implement the forward method
        # Step 1: Apply the convolution projection initialized in a previous step (expected shape: batch_size, embed_dim, H', W')
        # Step 2: Flatten spatial dimensions (parameter `start_dim=2`) (expected shape: batch_size, embed_dim, num_patches)
        # Step 3: Transpose to get output shape: (batch_size, num_patches, embed_dim)
        x = self.conv1(x) 
        x = self.flatten(x)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x


if __name__ == "__main__":
    batch_size, img_size, embed_dim = 2, 224, 768
    patch_emb = PatchEmbedding(224, 16, 3, embed_dim)
    x = torch.randn(batch_size, 3, img_size, img_size)
    patches = patch_emb(x)
    assert patches.shape == (batch_size, 196, embed_dim), "PatchEmbedding output shape incorrect"