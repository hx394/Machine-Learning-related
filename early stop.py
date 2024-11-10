from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=True,
    mode="min",
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="tft-best-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)

trainer = Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
    callbacks=[early_stop_callback, checkpoint_callback]
)
