import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import MetricCollection

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        image_size = 3 * 32 * 32
        classes = 10
        self.model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=image_size, out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, classes)
    )
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
        self.log('train_loss', loss, prog_bar=True)

        self.train_metrics.update(outputs, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        loss = self.loss(outputs, y)
        self.log('val_loss', loss)
        self.val_metrics(outputs, y)
        # self.log('val_accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=True) # Do not log every step 
        
    
    def test_step(self, batch):
        x, y = batch
        outputs = self.model(x)
        self.test_metrics(outputs, y)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer