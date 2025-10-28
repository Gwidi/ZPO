import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics import MetricCollection

class LitResNet18(L.LightningModule):
    def __init__(self, model, num_classes: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Use pre-trained ResNet18 model
        self.model = model

        # Change the last layer to match our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.loss = nn.CrossEntropyLoss()

        self.metrics = MetricCollection([
            torchmetrics.Accuracy(num_classes=num_classes, task="multiclass"),
            torchmetrics.F1Score(num_classes=num_classes, average='macro', task="multiclass"),
            torchmetrics.Precision(num_classes=num_classes, average='macro', task="multiclass"),
            torchmetrics.Recall(num_classes=num_classes, average='macro', task="multiclass")
        ])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        self.train_metrics.update(y_hat, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        self.val_metrics.update(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        self.test_metrics.update(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer