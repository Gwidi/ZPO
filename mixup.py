from torch.nn import functional as F
import torch 
from torchvision import datasets, transforms
import numpy as np
from lab04 import get_dataloaders
from matplotlib import pyplot as plt


class Mixup:
    """Applies Mixup to a batch of images and labels.

    Mixup constructs virtual training examples by forming convex combinations of pairs of examples and their labels.
    This technique helps to regularize the model and improve generalization. The object is called inside the training loop.

    Parameters
    ----------
    num_classes : int
        The total number of classes in the dataset.
    alpha : float, default=1.0
        Mixup hyperparameter for the Beta distribution. If alpha is 0,
        no mixup is applied.
    """
    def __init__(self, num_classes: int, alpha: float = 1.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        """Performs the Mixup transformation on a batch.

        Parameters
        ----------
        images : torch.Tensor
            A batch of images of shape (N, C, H, W).
        labels : torch.Tensor
            A batch of corresponding labels of shape (N,).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - **mixed_images** (*torch.Tensor*): The batch of mixed images.
            - **mixed_labels** (*torch.Tensor*): The batch of mixed, one-hot
              encoded labels.
        """
        if self.alpha <= 0:
            return images, F.one_hot(labels, num_classes=self.num_classes).float()
        
        batch_size, _, H, W = images.shape
        device = images.device

        # Create a random permutation of batch indices
        index = torch.randperm(batch_size, device=device)
        # Convert labels to one-hot encoding and cast to float
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()

        # Sample mixing lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        # Mix images
        mixed_images= lam * images + (1 - lam) * images[index]
        # Mix the one-hot labels
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]

        return mixed_images, mixed_labels

def main():
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

    ######################################################
    # Get a batch of images and labels
    images, labels = next(iter(train_dataloader))

    # Initialize and apply the Mixup transform
    mixup_transform = Mixup(num_classes=37, alpha=1.0)
    mixed_images, mixed_labels = mixup_transform(images, labels)

    # --- Visualization ---
    num_images_to_show = 4

    fig, axes = plt.subplots(1, num_images_to_show, figsize=(4 * num_images_to_show, 5))
    fig.suptitle(f"Mixup", fontsize=32)

    for i in range(num_images_to_show):
        # --- Plot the augmented image ---
        mixed_image_np = mixed_images[i].permute(1, 2, 0).numpy()
        # Unnormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mixed_image_np = mixed_image_np * std + mean
        # Normalize for display
        mixed_image_np = (mixed_image_np - mixed_image_np.min()) / (mixed_image_np.max() - mixed_image_np.min())

        axes[i].imshow(mixed_image_np)
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for the main title
    plt.show()


if __name__ == "__main__":
    main()