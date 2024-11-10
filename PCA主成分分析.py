import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 加载示例数据集
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 初始化 PCA 对象，并指定主成分数量（例如 2 个主成分）
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# 转换为 DataFrame，方便查看结果
pca_df = pd.DataFrame(pca_data, columns=['Principal Component 1', 'Principal Component 2'])
print(pca_df)

# 输出解释方差比例（每个主成分的方差占比）
print("Explained variance ratio:", pca.explained_variance_ratio_)
