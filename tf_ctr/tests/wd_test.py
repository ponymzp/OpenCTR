import unittest
import tensorflow as tf
import numpy as np
from tf_ctr.models.wd import WideAndDeep

# Wide 部分
wide_features_dim = 10  # 假设有 10 个宽特征

# Deep 部分
deep_sparse_features = [
    {'name': 'user_id', 'vocab_size': 10000},
    {'name': 'item_id', 'vocab_size': 5000},
]
deep_dense_features = ['age', 'rating']


# 初始化模型
model = WideAndDeep(
    wide_features_dim=wide_features_dim,
    deep_sparse_features=deep_sparse_features,
    deep_dense_features=deep_dense_features,
    embedding_dim=8,
    dnn_hidden_units=(128, 64),
    dropout=0.5
)

# 编译模型
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


import numpy as np

# Wide 部分输入数据
wide_X = np.random.rand(1000, wide_features_dim)

# Deep 部分 - 稀疏输入数据
deep_sparse_X = {
    'user_id_input': np.random.randint(0, 10000, size=(1000, 1)),
    'item_id_input': np.random.randint(0, 5000, size=(1000, 1))
}

# Deep 部分 - 稠密输入数据
deep_dense_X = {
    'age_input': np.random.rand(1000, 1) * 50,
    'rating_input': np.random.rand(1000, 1) * 5
}

# 合并所有输入
X = {'wide_input': wide_X, **deep_sparse_X, **deep_dense_X}
y = np.random.randint(0, 2, size=(1000, 1))  # 二分类标签


model.fit(X, y, batch_size=32, epochs=10)