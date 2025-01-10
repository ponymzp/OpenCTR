# -*- coding:utf-8 -*-
"""
Author:
    Pony Ma, ponymzp@163.com
Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C] https://arxiv.org/pdf/1606.07792.pdf
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout


class WideAndDeep(tf.keras.Model):
    def __init__(self,
                 deep_sparse_features,
                 deep_dense_features,
                 embedding_dim=8,
                 dnn_hidden_units=(128, 64),
                 dropout=0.5):
        """
        Wide and Deep 模型
        Args:
            deep_sparse_features: Deep 部分稀疏特征信息，字典形式
            deep_dense_features: Deep 部分稠密特征信息，列表形式
            embedding_dim: Embedding 层的维度
            dnn_hidden_units: DNN 隐藏层大小
            dropout: Dropout 概率
        """
        super(WideAndDeep, self).__init__()

        # Wide 部分
        self.wide_dense = Dense(1, activation=None, name="wide_dense")

        # Deep 部分 - 稀疏特征
        self.embedding_layers = {}
        for feat in deep_sparse_features:
            self.embedding_layers[feat['name']] = Embedding(
                input_dim=feat['vocab_size'],
                output_dim=feat['embedding_dim'],
                input_length=1,
                name=f"{feat['name']}_embedding"
            )

        # Deep 部分 - 全连接层
        self.dense_layers = []
        for units in dnn_hidden_units:
            self.dense_layers.append(Dense(units, activation="relu"))
            self.dense_layers.append(Dropout(dropout))

        self.deep_output_layer = Dense(1, activation=None, name="deep_output")

        # 特征名存储
        self.deep_sparse_features = deep_sparse_features
        self.deep_dense_features = deep_dense_features

    def call(self, inputs):
        """
        前向传播
        Args:
            inputs: 字典形式，包含 wide 输入、deep 稀疏输入、deep 稠密输入
        Returns:
            模型输出
        """
        # Wide 部分
        wide_input = inputs['wide_input']
        wide_output = self.wide_dense(wide_input)

        # Deep 部分 - 稀疏特征嵌入
        sparse_embeddings = []
        for feat in self.deep_sparse_features:
            sparse_input = inputs[f"{feat['name']}_input"]
            embedding = self.embedding_layers[feat['name']](sparse_input)
            sparse_embeddings.append(Flatten()(embedding))

        # Deep 部分 - 稠密特征
        dense_inputs = [inputs[f"{feat_name}_input"] for feat_name in self.deep_dense_features]

        # 拼接稀疏嵌入和稠密特征
        deep_input = Concatenate()(sparse_embeddings + dense_inputs)

        # Deep 部分 - 全连接网络
        deep_output = deep_input
        for layer in self.dense_layers:
            deep_output = layer(deep_output)
        deep_output = self.deep_output_layer(deep_output)

        # Wide 和 Deep 合并
        combined_output = tf.nn.sigmoid(wide_output + deep_output)
        return combined_output
