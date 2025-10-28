import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

class LitResNet18(L.LightningModule):
    def __init__(self, model, num_classes: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Use pre-trained ResNet18 model
        self.model = model

        # Change the last layer to match our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.loss = nn.CrossEntropyLoss()

        # Buffers to store running totals for correct predictions and total samples.
        self.register_buffer("train_correct", torch.tensor(0.0))
        self.register_buffer("train_total", torch.tensor(0.0))
        self.register_buffer("val_correct", torch.tensor(0.0))
        self.register_buffer("val_total", torch.tensor(0.0))
        self.register_buffer("test_correct", torch.tensor(0.0))
        self.register_buffer("test_total", torch.tensor(0.0))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Reset counters on the first batch of each training epoch
        if batch_idx == 0:
            self.train_correct = self.train_correct.new_zeros(1)
            self.train_total = self.train_total.new_zeros(1)

        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.train_correct += (preds == y).sum()
        self.train_total += y.size(0)
        
        acc = self.train_correct.float() / self.train_total
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Reset counters on the first batch of each validation epoch
        if batch_idx == 0:
            self.val_correct = self.val_correct.new_zeros(1)
            self.val_total = self.val_total.new_zeros(1)

        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.val_correct += (preds == y).sum()
        self.val_total += y.size(0)
        
        acc = self.val_correct.float() / self.val_total
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Reset counters on the first batch of the test run
        if batch_idx == 0:
            self.test_correct = self.test_correct.new_zeros(1)
            self.test_total = self.test_total.new_zeros(1)

        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.test_correct += (preds == y).sum()
        self.test_total += y.size(0)
        
        acc = self.test_correct.float() / self.test_total
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer