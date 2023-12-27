import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall

class LitSimpleCNN(pl.LightningModule):
    def __init__(self):
        super(LitSimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        
        #Linear layers
        self.fc1 = nn.Linear(in_features=128*49, out_features=49)
        
        # Test metrics
        self.test_accuracy = Accuracy(num_classes=2, average='macro', task='binary')
        self.test_precision = Precision(num_classes=2, average='macro', task='binary')
        self.test_recall = Recall(num_classes=2, average='macro', task='binary')
        
        # Validation metrics
        self.val_accuracy = Accuracy(num_classes=2, average='macro', task='binary')
        self.val_precision = Precision(num_classes=2, average='macro', task='binary')
        self.val_recall = Recall(num_classes=2, average='macro', task='binary')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.fc1(x))
        x = x.view(-1, 7, 7)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, labels)
        
        preds = torch.round(outputs)
        self.val_accuracy.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 0.001)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = torch.round(outputs)

        self.test_accuracy(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)

    def on_test_epoch_end(self):
        accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()

        print(f"Test Accuracy: {accuracy}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")