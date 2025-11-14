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
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

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
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
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

def postprocess_filter(img_tensor: torch.Tensor):
    """Helper function for post-processing and display."""
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    # Normalize to [0, 1] for display
    img = (img - img.min()) / (img.max() - img.min())
    return img


def visualize_filter(model, layer: str, filter_index: int, iterations: int = 50, lr: float = 0.1) -> np.ndarray:
    """Generates an image that maximally activates a specified filter."""
    # Start with a random noise image (our canvas)
    image = torch.randn(1, 3, 224, 224, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=lr, weight_decay=1e-6)

    # We need to hook into the model to capture the output of our target layer
    activation = None
    def hook(model, input, output):
        nonlocal activation
        activation = output

    handle = layer.register_forward_hook(hook)

    print(f"Computing Filter #{filter_index} of layer {layer.__class__.__name__}...")
    # Optimize the image to maximize the activation of the chosen filter
    for i in range(iterations):
        optimizer.zero_grad()
        # Forward pass to get the activation
        model(image)
        # Our "loss" is the negative of the mean activation of the chosen filter.
        # We negate it because optimizers minimize, but we want to maximize.
        loss = -torch.mean(activation[0, filter_index])
        loss.backward()
        optimizer.step()

    handle.remove() # Clean up the hook
    return postprocess_filter(image)

def task5():
    # Task 5: Visualize filters from a convolutional layer
    resnet34 = models.resnet34(weights="IMAGENET1K_V1")
    resnet34.eval()
    early_filter1 = visualize_filter(resnet34, resnet34.layer1[0].conv1, filter_index=0, iterations=50, lr=0.1)
    early_filter2 = visualize_filter(resnet34, resnet34.layer1[0].conv1, filter_index=1, iterations=50, lr=0.1)
    early_filter3 = visualize_filter(resnet34, resnet34.layer1[0].conv1, filter_index=2, iterations=50, lr=0.1)
    middle_filter1 = visualize_filter(resnet34, resnet34.layer3[0].conv2, filter_index=0, iterations=50, lr=0.1)
    middle_filter2 = visualize_filter(resnet34, resnet34.layer3[0].conv2, filter_index=1, iterations=50, lr=0.1)
    middle_filter3 = visualize_filter(resnet34, resnet34.layer3[0].conv2, filter_index=2, iterations=50, lr=0.1)
    late_filter1 = visualize_filter(resnet34, resnet34.layer4[1].conv1, filter_index=0, iterations=50, lr=0.1)
    late_filter2 = visualize_filter(resnet34, resnet34.layer4[1].conv1, filter_index=1, iterations=50, lr=0.1)
    late_filter3 = visualize_filter(resnet34, resnet34.layer4[1].conv1, filter_index=2, iterations=50, lr=0.1)

    
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    
    axs[0, 0].imshow(early_filter1)
    axs[0, 0].set_title("Early Layer - Filter 0")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(early_filter2)
    axs[0, 1].set_title("Early Layer - Filter 1")
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(early_filter3)
    axs[0, 2].set_title("Early Layer - Filter 2")
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(middle_filter1)
    axs[1, 0].set_title("Middle Layer - Filter 0")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(middle_filter2)
    axs[1, 1].set_title("Middle Layer - Filter 1")
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(middle_filter3)
    axs[1, 2].set_title("Middle Layer - Filter 2")
    axs[1, 2].axis('off')
    
    axs[2, 0].imshow(late_filter1)
    axs[2, 0].set_title("Late Layer - Filter 0")
    axs[2, 0].axis('off')
    
    axs[2, 1].imshow(late_filter2)
    axs[2, 1].set_title("Late Layer - Filter 1")
    axs[2, 1].axis('off')
    
    axs[2, 2].imshow(late_filter3)
    axs[2, 2].set_title("Late Layer - Filter 2")
    axs[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def task6():
    resnet34 = models.resnet34(weights="IMAGENET1K_V1")
    resnet34.eval()

    # Load the labels for ImageNet
    labels_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    response = requests.get(labels_url)
    labels = response.json()
    labels = {int(k):v[1] for k,v in labels.items()}

    img = Image.open("data/Kot-bengalski-brazowy.jpg")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformed_img = transform(img)
    input = transform_normalize(transformed_img).unsqueeze(0)
    prediction = resnet34(input).squeeze(0).softmax(0)
    predicted_label_idx = prediction.argmax().item()
    predicted_label = labels[predicted_label_idx]
    prediction_score = prediction[predicted_label_idx].item()

    print(f'Predicted: {predicted_label}, ({prediction_score:.2f})')

    integrated_gradients = IntegratedGradients(resnet34)
    attributions_ig = integrated_gradients.attribute(input, target=predicted_label_idx, n_steps=200)

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)


if __name__ == "__main__":
    #main()
    #task4()
   task6()