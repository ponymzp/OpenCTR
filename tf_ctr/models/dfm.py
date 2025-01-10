# -*- coding:utf-8 -*-
"""
DeepFM模型
Author:
    Pony Ma, ponymzp@163.com
References
    - [DeepFM](https://www.ijcai.org/proceedings/2017/0239.pdf)

"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout, Concatenate
from tensorflow.keras.regularizers import l2


class DeepFM(tf.keras.Model):
    def __init__(
        self,
        dense_features,
        sparse_features,
        hidden_units=(128, 64),
        regularization=0.001,
        dropout_rate=0.5
    ):
        """
        初始化模型
        Args:
            dense_features: 稠密特征的名称列表。
            sparse_features: 稀疏特征的配置信息，包含特征名和词汇表大小。
            hidden_units: 深度部分的隐藏层单元数列表。
            regularization: 正则化系数。
            dropout_rate: Dropout比例。
        """
        super(DeepFM, self).__init__()
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.hidden_units = hidden_units
        self.regularization = regularization
        self.dropout_rate = dropout_rate

        # 稠密特征归一化层
        self.dense_normalization_layers = {
            name: tf.keras.layers.LayerNormalization(axis=-1) for name in self.dense_features
        }

        # Embedding 层
        self.embedding_layers = {
            feat['name']: Embedding(
                input_dim=feat['vocab_size'],
                output_dim=feat['embedding_dim'],
                input_length=1,
                name=f"{feat['name']}_embedding"
            ) for feat in self.sparse_features
        }

        # FM部分的一阶线性层
        self.linear_layer = Dense(1, activation=None, kernel_regularizer=l2(self.regularization))

        # 深度部分的隐藏层
        self.deep_layers = []
        for units in hidden_units:
            self.deep_layers.append(Dense(units, activation="relu", kernel_regularizer=l2(regularization)))
            self.deep_layers.append(Dropout(dropout_rate))

        # 输出层
        self.output_layer = Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=l2(regularization)
        )

    def call(self, inputs):
        """
        前向传播
        Args:
            inputs: 包含 dense 和 sparse 特征的字典。
        Returns:
            模型的输出。
        """
        # 处理稠密特征
        dense_normalized_features = [
            self.dense_normalization_layers[name](inputs[name])
            for name in self.dense_features
        ]

        # 稀疏特征嵌入并展平
        sparse_embeddings = [
            self.embedding_layers[feat['name']](inputs[feat['name']])
            for feat in self.sparse_features
        ]

        # 合并稠密特征和稀疏特征嵌入
        sparse_embeddings_flattened = [Flatten()(embedding) for embedding in sparse_embeddings]
        combined_features = Concatenate()(dense_normalized_features + sparse_embeddings_flattened)

        # FM 部分：一阶线性部分
        linear_terms = self.linear_layer(combined_features)

        # FM 部分：二阶交互部分
        summed_embeddings = tf.reduce_sum(sparse_embeddings, axis=0)
        summed_embeddings_square = tf.square(summed_embeddings)

        squared_embeddings = [tf.square(embedding) for embedding in sparse_embeddings]
        squared_sum_embeddings = tf.reduce_sum(squared_embeddings, axis=0)

        cross_terms = 0.5 * tf.reduce_sum(summed_embeddings_square - squared_sum_embeddings, axis=-1, keepdims=True)
        cross_terms = Flatten()(cross_terms)

        # 深度部分
        x_deep = combined_features
        for layer in self.deep_layers:
            x_deep = layer(x_deep)

        # 输出
        output = self.output_layer(linear_terms + cross_terms + x_deep)
        return output