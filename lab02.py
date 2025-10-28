import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from LitResNet18 import LitResNet18


def main():
    '''Exercise 1: Load the OxfordIIITPet dataset and create DataLoaders for training, validation, and testing.'''
    # Define transformations
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define augmentations for training data (5 different) - task 5 
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the OxfordIIITPet dataset
    train_dataset = OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
    test_dataset = OxfordIIITPet(root='./data', split='test', download=True, transform=transform)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    train_OxfordIIITPet, val_OxfordIIITPet = torch.utils.data.random_split(train_dataset, [0.75, 0.25])
    # Apply augmentations only to training dataset
    train_OxfordIIITPet.dataset.transform = augmentations

    train_loader = torch.utils.data.DataLoader(train_OxfordIIITPet, batch_size=64, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_OxfordIIITPet, batch_size=64, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=2)


    # Load ResNet-18 model
    resnet18 = models.resnet18(weights="IMAGENET1K_V1")
    resnet18.fc = nn.Linear(resnet18.fc.in_features, 37)

    # Initialize the model and trainer
    model = LitResNet18(resnet18, num_classes=37)

    # Exercise 2 Train the model and verify its performance on the test set.
    trainer = L.Trainer(max_epochs=10, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()