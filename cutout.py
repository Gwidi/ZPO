import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from lab04 import get_dataloaders

class Cutout:
    """Applies Cutout augmentation to a batch of images.

    This augmentation randomly masks out one or more square regions of an input image.

    Parameters
    ----------
    n_holes : int
        Number of patches to cut out from an image.
    length : int
        The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, image: torch.Tensor):
        """Applies the cutout transformation.

        Parameters
        ----------
        image : torch.Tensor
            Tensor image of size (C, H, W).

        Returns
        -------
        torch.Tensor
            Image with `n_holes` of dimension `length` x `length` cut out.
        """
        ############# TODO: Student code #####################
        # Step 1: Initialize a single channel mask with all ones (`np.ones`) and input tensor HxW dimensions
        mask = np.ones((image.shape[1], image.shape[2]))

        # Create n_holes square patches in the mask
        for _ in range(self.n_holes):
            # Step 2: Randomly select center point for the hole
            y = np.random.randint(image.shape[1])
            x = np.random.randint(image.shape[2])
            # Step 3: Calculate boundaries of the square patch, considering (y, x) as square patch center and `self.length` as patch length
            y1 = y - self.length // 2
            y2 = y + self.length // 2
            x1 = x - self.length // 2
            x2 = x + self.length // 2
            # Step 4: Clip boundaries of the square patch to image boundaries `np.clip`
            y1 = np.clip(y1, 0, image.shape[1])
            y2 = np.clip(y2, 0, image.shape[1])
            x1 = np.clip(x1, 0, image.shape[2])
            x2 = np.clip(x2, 0, image.shape[2])

            # Step 5: Set the patch region in mask to 0 (black out the region)
            mask[y1:y2, x1:x2] = 0.0

        # Step 6: Convert numpy mask to torch tensor
        mask = torch.from_numpy(mask).to(image.device)
        # Step 7: Add a channel dimension to the mask: (H, W) => (C, H, W)
        mask = mask.unsqueeze(0).expand_as(image)
        # Step 8: Apply mask to image (element-wise multiplication)
        image = image * mask

        ######################################################

        return image

def main():
    ############# TODO: Student code #####################
    # Add above `get_dataloaders` function and call it to get `train_dataloader`
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

    ######################################################
    # Get a batch of images and labels
    images, labels = next(iter(train_dataloader))

    # Initialize and apply the Cutout transform
    cutout_transform = Cutout(n_holes=1, length=64)
    augmented_images = images.clone()
    for i in range(len(images)):
        augmented_images[i] = cutout_transform(images[i])

    # --- Visualization ---
    num_images_to_show = 4

    fig, axes = plt.subplots(1, num_images_to_show, figsize=(4 * num_images_to_show, 5))
    fig.suptitle(f"Cutout", fontsize=32)

    for i in range(num_images_to_show):
        # --- Plot the augmented image ---
        cutout_image_np = augmented_images[i].permute(1, 2, 0).numpy()
        # Unnormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        cutout_image_np = cutout_image_np * std + mean
        # Normalize for display
        cutout_image_np = (cutout_image_np - cutout_image_np.min()) / (cutout_image_np.max() - cutout_image_np.min())
        axes[i].imshow(cutout_image_np)
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for the main title
    plt.show()


if __name__ == "__main__":
    main()
