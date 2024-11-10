# 导入库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. 加载并预处理数据
# 假设数据在 data.csv 中，有多列 'Open'、'High'、'Low'、'Close'、'Volume'
data = pd.read_csv('data.csv')
features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

# 标准化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# 创建时间序列数据
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length, 3])  # 使用 'Close' 列（索引 3）作为预测目标
    return np.array(sequences), np.array(labels)

seq_length = 60  # 使用前 60 天的数据预测下一天
X, y = create_sequences(scaled_features, seq_length)

# 转换为 Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# 2. 定义 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 3. 设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

# 4. 训练模型
model.train()
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. 测试模型
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        predictions.append(output.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

# 转换为数组
predictions = np.concatenate(predictions).reshape(-1, 1)
actuals = np.concatenate(actuals).reshape(-1, 1)

# 反向标准化
predictions = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], features.shape[1]-1)), predictions]))[:, -1]
actuals = scaler.inverse_transform(np.hstack([np.zeros((actuals.shape[0], features.shape[1]-1)), actuals]))[:, -1]

# 6. 可视化结果
plt.figure(figsize=(10,6))
plt.plot(actuals, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# 7. 保存模型（可选）
torch.save(model.state_dict(), 'stock_gru_model.pth')