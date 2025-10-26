import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models
import torchmetrics
from torchmetrics import MetricCollection

class LitResNet18Pretrained(L.LightningModule):
    def __init__(self, num_classes=2, pretrained=True, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Use pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=pretrained)

        # Change the last layer to match our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.loss = nn.CrossEntropyLoss()

        # Metrics
        task = "multiclass" if num_classes > 2 else "binary"
        self.metrics = MetricCollection([
            torchmetrics.Accuracy(num_classes=num_classes, task=task),
            torchmetrics.F1Score(num_classes=num_classes, average='macro', task=task)
        ])

        if task == "binary":
            self.metrics = MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Recall(task="binary")
            ])
        else:
            # Multiclass metrics z num_classes
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

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        if self.hparams.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(positive class)
            self.train_metrics.update(probs, y)
        else:
            self.train_metrics.update(logits, y)
    
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        if self.hparams.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(positive class)
            self.val_metrics.update(probs, y)
        else:
            self.val_metrics.update(logits, y)
    
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch):
        x, y = batch
        logits = self.model(x)
        
        if self.hparams.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(positive class)
            self.test_metrics.update(probs, y)
        else:
            self.test_metrics.update(logits, y)
    
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer