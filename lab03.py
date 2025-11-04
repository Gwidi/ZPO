'''Laboratory 3: Model interpretability, filters and features visualization'''

import json

from matplotlib import pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image

def main():
    # Task 1
    resnet34 = models.resnet34(weights="IMAGENET1K_V1")
    resnet34.eval()

    # Load the labels for ImageNet
    labels_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    response = requests.get(labels_url)
    labels = response.json()
    labels = {int(k):v[1] for k,v in labels.items()} # Convert keys to integers and get the label name

    # Task 3: Load an image from a URL and write infercence code 
    img_url = "https://images.pexels.com/photos/20816519/pexels-photo-20816519.jpeg"
    #img = Image.open(requests.get(img_url, stream=True).raw)

    # Test on different images
    img = Image.open("data/Kot-bengalski-brazowy.jpg")
    
    batch = preprocess_image(img)
    prediction = resnet34(batch).squeeze(0).softmax(0)
    predicted_label = labels[prediction.argmax().item()]
    prediction_score = prediction[prediction.argmax().item()].item()

    fig = plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.title(f"Predicted class: '{predicted_label}' with confidence {prediction_score:.2f}")
    plt.axis('off')
    plt.show()



def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocesses an image for model inference."""
    # Task 2 : write a model input transformation pipeline
    # ImageNet-based pre-processing image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0) # Add batch dimension
    return input_tensor


def postprocess(output: torch.Tensor, labels: dict[int, str]) -> tuple[str, int, float]:
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = labels[pred_label_idx.item()]
    return predicted_label, pred_label_idx, prediction_score.item()



if __name__ == "__main__":
    main()