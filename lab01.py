import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
import lightning as L
from LitModel import LitModel
from datetime import datetime
from LitModelSimpleNN import LitModelSimpleNN
from LitModelSimpleCNN import LitModelSimpleCNN
from CatsvsDogs import CatsvsDogs
from torch.utils.data import random_split, DataLoader
from LitResNet18Pretrained import LitResNet18Pretrained

def main():
    convert_to_tensor = torchvision.transforms.ToTensor()

    train_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', download=True, train=True, transform=convert_to_tensor)
    test_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', train=False, transform=convert_to_tensor)

    print("There are", len(train_cifar10), "images in the training set.")
    print("There are", len(test_cifar10), "images in the test set.")
    print("Each image is of size", train_cifar10[0][0].shape)

    cifar10_classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    visualize = False

    figure = plt.figure(figsize=(8, 4))
    cols, rows = 5, 2
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_cifar10), size=(1,)).item()
        img, label = train_cifar10[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(cifar10_classes[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))  # CHW -> HWC
    if visualize:
        plt.show()

    train_cifar10, val_cifar10 = torch.utils.data.random_split(train_cifar10, [0.75, 0.25])

    train_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=64, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_cifar10, batch_size=64, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=64, num_workers=2)

    # Define a simple feedforward neural network
    image_size = 3 * 32 * 32

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=image_size, out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, len(cifar10_classes))
    )

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss() # Cross-entropy loss has softmax built-in

    device = torch.device("cpu")
    model = model.to(device)

    # Variables for tracking best accuracy
    best_val_accuracy = 0.0
    best_model_path = "best_model.pt"
    patience = 3
    patience_counter = 0

    # Create a catalogue to save the checkpoints
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(10):  # przejdźmy po naszym zbiorze uczącym 10 razy
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            # load inputs and labels
            inputs, labels = data

            # move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward pass and optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # printing statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10th mini-batch
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        model.eval()
        #Evaluate the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f'Epoch accuracy of the model on the validation images: {val_acc} %%')
        
        # Save the model if it has the best accuracy so far
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            # Save the best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_accuracy
            }
            torch.save(checkpoint, f"checkpoints/{best_model_path}")
            print(f'✓ New best model saved! Validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print('Finished Training')

    # Load the best model for testing
    if os.path.exists(f"checkpoints/{best_model_path}"):
        checkpoint = torch.load(f"checkpoints/{best_model_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

def exercise_8():
    convert_to_tensor = torchvision.transforms.ToTensor()

    train_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', download=True, train=True, transform=convert_to_tensor)
    test_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', train=False, transform=convert_to_tensor)

    model = LitModel()

    train_cifar10, val_cifar10 = torch.utils.data.random_split(train_cifar10, [0.75, 0.25])

    cifar10_classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=64, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_cifar10, batch_size=64, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=64, num_workers=2)

    trainer  = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)

def exercise_9():
    convert_to_tensor = torchvision.transforms.ToTensor()

    train_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', download=True, train=True, transform=convert_to_tensor)
    test_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', train=False, transform=convert_to_tensor)

    model = LitModelSimpleNN()

    train_cifar10, val_cifar10 = torch.utils.data.random_split(train_cifar10, [0.75, 0.25])

    cifar10_classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=64, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_cifar10, batch_size=64, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=64, num_workers=2)

    trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)

def exercise_10():
    convert_to_tensor = torchvision.transforms.ToTensor()

    train_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', download=True, train=True, transform=convert_to_tensor)
    test_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', train=False, transform=convert_to_tensor)

    model = LitModelSimpleCNN()

    train_cifar10, val_cifar10 = torch.utils.data.random_split(train_cifar10, [0.75, 0.25])

    train_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=64, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_cifar10, batch_size=64, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=64, num_workers=2)

    trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)

def exercise_11():
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
])

    dataset = CatsvsDogs('CatsVsDogs/PetImages', transform=transform)
    print("Class counts:", dataset.get_class_counts())

    # Split the dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = LitModelSimpleCNN(num_classes=2)

    trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)

def exercise_12():
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    ])
    dataset = CatsvsDogs('CatsVsDogs/PetImages', transform=transform)

    # Split the dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = LitResNet18Pretrained(num_classes=2, pretrained=True)

    trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    exercise_12()

