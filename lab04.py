'''Advanced classification with Transformer network'''

import torch
import lightning as L
from torchvision import datasets, transforms

def get_dataloaders(batch_size: int = 32, num_workers: int = 4):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the OxfordIIITPet dataset
    trainval_dataset = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
    test_dataset = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)

    ############# TODO: Student code #####################

    # 1. Split train dataset into train and validation
    train, val = torch.utils.data.random_split(trainval_dataset, [0.75, 0.25])

    # 2. Create data loaders
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    ######################################################

    return train_dataloader, val_dataloader, test_dataloader



if __name__ == '__main__':
    task1()