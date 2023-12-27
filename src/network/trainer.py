from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.0001,
   patience=3,
   verbose=True,
   mode='min'
)

trainer = pl.Trainer(callbacks=[early_stop_callback])