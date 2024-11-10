import optuna
from pytorch_lightning import Trainer
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# Define objective function for Optuna
def objective(trial):
    # Suggest values for each hyperparameter
    hidden_size = trial.suggest_int("hidden_size", 16, 64)
    attention_head_size = trial.suggest_int("attention_head_size", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 64)

    # Instantiate TFT model with suggested parameters
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        output_size=7,  # For quantile loss
        reduce_on_plateau_patience=4,
    )

    # Set up trainer with early stopping to prevent overfitting
    trainer = Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        gradient_clip_val=0.1,
    )

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Evaluate performance on validation set
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss

# Run Optuna study to optimize the hyperparameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print best trial
print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
