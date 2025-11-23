import lightning as L
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchmetrics import MetricCollection, Precision, Recall, F1Score, Accuracy
from lab04 import get_dataloaders

from transformer import VisionTransformer


class LitVisionTransformer(L.LightningModule):
    """Lightning module for Vision Transformer."""
    def __init__(self, num_classes: int, lr: float = 1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters() # Save hyperparameters

        self.vision_transformer = VisionTransformer(num_classes=num_classes, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        metrics = MetricCollection({
            'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
            'precision': Precision(task="multiclass", num_classes=num_classes, average='macro'),
            'recall': Recall(task="multiclass", num_classes=num_classes, average='macro'),
            'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro')
        })
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.vision_transformer(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        
        # Update and log metrics
        self.train_metrics(outputs, labels)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        
        # Update and log metrics
        self.val_metrics(outputs, labels)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss)
        
        # Update and log metrics
        self.test_metrics(outputs, labels)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


def main():
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(batch_size=32, num_workers=4)
    # Instantiate the Lightning module
    # num_classes should be 37 for the OxfordIIITPet dataset
    model = LitVisionTransformer(num_classes=37, lr=1e-3)

    # Instantiate a Lightning Trainer
    trainer = L.Trainer(max_epochs=10, accelerator='auto')

    # Start the training process
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()