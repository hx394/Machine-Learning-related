import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import load_iris

# 加载示例数据集
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 计算每个特征的 VIF
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# 输出 VIF 结果
vif_df = calculate_vif(df)
print(vif_df)
