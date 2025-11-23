import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from lab04 import get_dataloaders


class Cutmix:
    """Applies Cutmix augmentation to a batch of images and labels.

    Cutmix is a data augmentation technique that involves cutting a patch from one image and pasting it onto another.
    The labels are then mixed proportionally to the area of the patch. The object is called inside the training loop.

    Parameters
    ----------
    alpha : float, default=1.0
        Cutmix hyperparameter for the Beta distribution. This controls the
        distribution of the patch sizes. If alpha is 0, no Cutmix is applied.
    p : float, default=0.5
        The probability of applying the Cutmix augmentation to each sample in a batch.
    """
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p

    def _rand_bbox(self, size, lam):
        """Generates a random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample the center of the box
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Calculate box coordinates, clipping to be within image boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        """Performs the Cutmix transformation on a batch.

        This method applies Cutmix to each sample in the batch with a probability `self.p`.

        Parameters
        ----------
        images : torch.Tensor
            A batch of images of shape (N, C, H, W).
        labels : torch.Tensor
            A batch of corresponding labels of shape (N,).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - **mixed_images** (*torch.Tensor*): The batch with Cutmix applied.
            - **labels_a** (*torch.Tensor*): The original labels.
            - **labels_b** (*torch.Tensor*): The labels from the shuffled batch corresponding to the pasted patches.
            - **lam_batch** (*torch.Tensor*): A tensor of shape (N,) containing the mixing ratio for each sample.
                                            `lam_batch` is 1.0 for samples where Cutmix was not applied.
        """
        # Return the original batch if augmentation is disabled
        if self.alpha <= 0 or self.p <= 0:
            lam = torch.ones(images.size(0), device=images.device)
            return images, labels, labels, lam

        batch_size, _, H, W = images.shape
        device = images.device

        # Get a shuffled batch for mixing
        index = torch.randperm(batch_size, device=device)
        labels_a, labels_b = labels, labels[index]

        # Initialize outputs
        mixed_images = images.clone()
        lam_batch = torch.ones(batch_size, device=device)

        # Iterate over each sample in the batch
        for i in range(batch_size):
            # Apply Cutmix with probability p
            if torch.rand(1).item() < self.p:
                # 1. Sample lambda from the Beta distribution
                lam = np.random.beta(self.alpha, self.alpha)

                # 2. Generate the bounding box for the patch
                bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)

                # 3. Get the partner image to cut the patch from
                partner_index = index[i]

                # 4. Paste the patch onto the original image
                mixed_images[i, :, bby1:bby2, bbx1:bbx2] = images[partner_index, :, bby1:bby2, bbx1:bbx2]

                # 5. Adjust lambda to match the true patch area and store it
                area = (bbx2 - bbx1) * (bby2 - bby1)
                lam_adjusted = 1.0 - (area / (H * W))
                lam_batch[i] = lam_adjusted

        return mixed_images, labels_a, labels_b, lam_batch

def main():
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

    ######################################################
    # Get a batch of images and labels
    images, labels = next(iter(train_dataloader))

    # Initialize and apply the Cutmix transform
    cutmix_transform = Cutmix(alpha=1.0)
    augmented_images, labels_a, labels_b, lam_batch = cutmix_transform(images, labels)

    # --- Visualization ---
    num_images_to_show = 4

    fig, axes = plt.subplots(1, num_images_to_show, figsize=(4 * num_images_to_show, 5))
    fig.suptitle(f"Cutmix", fontsize=32)

    for i in range(num_images_to_show):
        lam = lam_batch[i]
        # --- Plot the augmented image ---
        mixed_image_np = augmented_images[i].permute(1, 2, 0).numpy()
        # Unnormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mixed_image_np = mixed_image_np * std + mean
        # Normalize for display
        mixed_image_np = (mixed_image_np - mixed_image_np.min()) / (mixed_image_np.max() - mixed_image_np.min())

        axes[i].imshow(mixed_image_np)
        axes[i].set_title(f"Labels: {labels_a[i]} & {labels_b[i]} (λ ≈ {lam:.2f})")
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for the main title
    plt.show()


if __name__ == "__main__":
    main()