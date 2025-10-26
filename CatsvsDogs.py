from torch.utils.data import Dataset
import torch
import torchvision
import os
import glob
from PIL import Image

def read_image(img_path):
    return Image.open(img_path).convert('RGB')


class CatsvsDogs(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Path to the main directory with subdirectories 'cats' and 'dogs'
            transform (callable, optional): Optional transformations
        """
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        cats_dir = os.path.join(data_dir, 'Cat')
        if os.path.exists(cats_dir):
             cat_images = glob.glob(os.path.join(cats_dir, '*.jpg'))
             self.image_paths.extend(cat_images)
             self.labels.extend([0] * len(cat_images))  

        dogs_dir = os.path.join(data_dir, 'Dog')
        if os.path.exists(dogs_dir):
            dog_images = glob.glob(os.path.join(dogs_dir, '*.jpg'))
            self.image_paths.extend(dog_images)
            self.labels.extend([1] * len(dog_images))

        print(f"Loaded {len(self.image_paths)} images: {len([l for l in self.labels if l == 0])} cats, {len([l for l in self.labels if l == 1])} dogs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = read_image(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
        
    def get_class_counts(self):
        """Returns the number of images per class."""
        cats = sum(1 for label in self.labels if label == 0)
        dogs = sum(1 for label in self.labels if label == 1) 
        return {'cats': cats, 'dogs': dogs}