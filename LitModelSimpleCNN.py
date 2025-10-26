import torch
import torch.nn as nn
import torchvision
import lightning as L
import torchmetrics
from torchmetrics import MetricCollection

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class LitModelSimpleCNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        classes = 10
        self.model = SimpleCNN()
        self.loss = nn.CrossEntropyLoss()
        self.metrics = MetricCollection([
            torchmetrics.Accuracy(num_classes=classes, task="multiclass"),
            torchmetrics.F1Score(num_classes=classes, average='macro', task="multiclass"),
            torchmetrics.Precision(num_classes=classes, average='macro', task="multiclass"),
            torchmetrics.Recall(num_classes=classes, average='macro', task="multiclass")
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')
    
    def training_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = self.loss(outputs, y)
        #self.log('train_loss', loss, prog_bar=True)
        self.train_metrics.update(outputs, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = self.loss(outputs, y)
        self.log('val_loss', loss)
        self.val_metrics.update(outputs, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True) # Do not log every step 
        
    
    def test_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        self.test_metrics.update(outputs, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    