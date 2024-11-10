# 读取包含多个特征的数据
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
