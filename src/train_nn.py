import torch
from src.network.basic_cnn import LitSimpleCNN
from src.network.trainer import trainer
from torch.utils.data import DataLoader


train_dataset = torch.load("datasets/train_dataset.pt")
val_dataset = torch.load("datasets/val_dataset.pt")
test_dataset = torch.load("datasets/test_dataset.pt")

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = LitSimpleCNN()
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model, test_loader)

trainer.save_checkpoint("saved_models/model_checkpoint.ckpt")