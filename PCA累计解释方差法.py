import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载示例数据并标准化
data = load_iris()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.data)

# 初始化 PCA 对象并拟合数据
pca = PCA()
pca.fit(scaled_data)

# 累积解释方差
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 绘制累积解释方差图
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.show()
