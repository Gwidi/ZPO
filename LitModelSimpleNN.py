import lightning as L
import torch
import torchmetrics
import torch.nn as nn
from torchmetrics import MetricCollection

class SimpleNN(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_classes=10):
        super().__init__()
        
        # Warstwa spłaszczająca
        self.flatten = nn.Flatten()
        
        # Jedna współdzielona warstwa aktywacji
        self.relu = nn.ReLU(inplace=True)
        
        # Warstwy w pełni połączone
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x ma wymiary: [batch_size, 3, 32, 32]
        
        x = self.flatten(x)    # [batch_size, 3072]
        
        x = self.fc1(x)        # [batch_size, 512]
        x = self.relu(x)
        
        x = self.fc2(x)        # [batch_size, 256]
        x = self.relu(x)
        
        x = self.fc3(x)        # [batch_size, 128]
        x = self.relu(x)
        
        x = self.fc4(x)        # [batch_size, 10]
        
        return x

class LitModelSimpleNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        image_size = 3 * 32 * 32
        classes = 10
        self.model = SimpleNN()
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
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self.loss(outputs, y)
        self.log('train_loss', loss, prog_bar=True)

        self.train_metrics.update(outputs, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        loss = self.loss(outputs, y)
        self.log('val_loss', loss)
        self.val_metrics.update(outputs, y)
        # self.log('val_accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=True) # Do not log every step 
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        self.test_metrics.update(outputs, y)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer