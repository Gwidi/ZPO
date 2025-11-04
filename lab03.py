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
from torchviz import make_dot


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

def task4():
    # Task 4: Visualize the model architecture using torchviz
    resnet34 = models.resnet34(weights="IMAGENET1K_V1")
    resnet34.eval()
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    # Perform a forward pass to get the output tensor
    output = resnet34(dummy_input)

    # Generate the graph visualization
    # We visualize the output tensor and specify the model's parameters for a clearer graph.
    dot = make_dot(output, params=dict(resnet34.named_parameters()))

    # Save the graph to a file (e.g., PDF, PNG)
    dot.render("resnet34_torchvision_graph", format="png", cleanup=True)



if __name__ == "__main__":
    #main()
    task4()