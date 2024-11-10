import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Data Preparation

# Load data and extract features (assuming CSV has columns like Open, High, Low, Close, Volume)
data = pd.read_csv('data.csv')
features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Define function to create sequences
def create_sequences(data, seq_length, pred_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length - pred_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length:i+seq_length+pred_length, 3])  # 'Close' as target
    return np.array(sequences), np.array(labels)

seq_length = 60  # Input sequence length (e.g., 60 days)
pred_length = 5  # Prediction length (e.g., predict next 5 days)

X, y = create_sequences(scaled_features, seq_length, pred_length)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split data into training and testing sets
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# 2. Transformer Model with Learnable Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, embed_dim))

    def forward(self, x):
        return x + self.positional_encoding


class TransformerStockPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, seq_length, pred_length):
        super(TransformerStockPredictor, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, seq_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction layer
        self.fc_out = nn.Linear(embed_dim, pred_length)

    def forward(self, x):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Transformer encoder
        transformer_out = self.transformer_encoder(x)

        # Take the last time step for prediction
        out = self.fc_out(transformer_out[:, -1, :])
        return out

# Model Parameters
input_dim = 5          # Number of input features (e.g., Open, High, Low, Close, Volume)
embed_dim = 64         # Embedding dimension
num_heads = 4          # Number of attention heads
num_layers = 3         # Number of transformer encoder layers
pred_length = 5        # Prediction length (predict next 5 days)

# Instantiate model, loss, and optimizer
model = TransformerStockPredictor(input_dim, embed_dim, num_heads, num_layers, seq_length, pred_length).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. Training and Evaluation Loops

def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
            output = model(X_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    return predictions, actuals


# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, epochs=50)
predictions, actuals = evaluate_model(model, test_loader)

# 4. Inverse Transform and Plot Results
# Reshape predictions and actuals to match original scale
predictions = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], features.shape[1]-1)), predictions]))[:, -pred_length:]
actuals = scaler.inverse_transform(np.hstack([np.zeros((actuals.shape[0], features.shape[1]-1)), actuals]))[:, -pred_length:]

# Plot results
plt.figure(figsize=(12, 6))
for i in range(pred_length):
    plt.plot(range(i, i + len(predictions)), actuals[:, i], label=f"Actual Day {i+1}")
    plt.plot(range(i, i + len(predictions)), predictions[:, i], linestyle='--', label=f"Predicted Day {i+1}")

plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
