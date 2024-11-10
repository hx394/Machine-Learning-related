#pip install torch pytorch-lightning pytorch-forecasting

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Load and Prepare Data

# Load the dataset (assuming it has 'Open', 'High', 'Low', 'Close', 'Volume', and 'Date' columns)
data = pd.read_csv('data.csv', parse_dates=['Date'])
data['time_idx'] = (data['Date'] - data['Date'].min()).dt.days  # create a time index
data['month'] = data['Date'].dt.month  # add month as a feature
data['year'] = data['Date'].dt.year  # add year as a feature

# Normalize continuous features
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# 2. Define Parameters for TFT Model
max_encoder_length = 60  # past sequence length (e.g., 60 days of history)
max_prediction_length = 5  # forecast horizon (e.g., predict next 5 days)

# Define the dataset for TFT
training_cutoff = data["time_idx"].max() - max_prediction_length

# 3. Create TimeSeriesDataSet for TFT
# This dataset handles the temporal and static covariates and creates samples for the TFT model
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close",
    group_ids=["year"],  # Grouping on 'year' or other unique identifiers
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx", "month", "year"],  # known features (e.g., date)
    time_varying_unknown_reals=["Open", "High", "Low", "Close", "Volume"],  # unknown features that change over time
    static_categoricals=["year"],  # static feature (e.g., stock ticker, company sector)
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)

# Create data loaders for training and validation
train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=4)
val_dataloader = validation.to_dataloader(train=False, batch_size=32, num_workers=4)

# 4. Define the TFT Model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,  # Hidden size of the network
    attention_head_size=4,  # Number of attention heads
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,  # Number of quantiles to predict
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4,
)

# Count the number of parameters (for reference)
print(f"Number of parameters in the model: {tft.size()/1e3:.1f}k")

# 5. Train the Model
trainer = Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=0.1,
)

# Fit the model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# 6. Evaluate and Forecast

# Raw predictions
raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)

# Helper function to plot predictions
def plot_prediction(x, raw_predictions, idx=0, quantile=0.5):
    """
    Plots a single time series prediction for interpretability.
    Args:
    x: Time series input data
    raw_predictions: TFT model raw predictions
    idx: Index of the prediction to plot
    quantile: Quantile of the prediction to display (0.5 for median)
    """
    encoder_target = x["encoder_target"][idx].detach().cpu().numpy()
    decoder_target = x["decoder_target"][idx].detach().cpu().numpy()
    prediction = raw_predictions["prediction"][idx].detach().cpu().numpy()

    plt.plot(range(len(encoder_target)), encoder_target, label="Historical")
    plt.plot(
        range(len(encoder_target), len(encoder_target) + len(decoder_target)),
        decoder_target,
        label="True Future",
    )
    plt.plot(
        range(len(encoder_target), len(encoder_target) + len(decoder_target)),
        prediction[..., quantile],
        label=f"Predicted (q={quantile})",
    )
    plt.legend()
    plt.show()

# Plot example predictions
plot_prediction(x, raw_predictions, idx=0, quantile=0.5)

# 7. Model Interpretation

# Get feature importances
interpretation = tft.interpret_output(raw_predictions, reduction="sum")

# Plot attention and feature importances
tft.plot_interpretation(interpretation)
