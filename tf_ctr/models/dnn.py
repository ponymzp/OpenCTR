# -*- coding:utf-8 -*-
"""
DNN模型
Author:
    Pony Ma, ponymzp@163.com
"""
import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Embedding, Concatenate

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.regularizers import l2

class DNN(tf.keras.Model):
    def __init__(
            self,
            dense_features,
            sparse_features,
            dnn_hidden_units=(128, 64),
            regularization=0.001
      ):
        """
        初始化模型
        Args:
            dense_features: 稠密特征的名称列表。
            sparse_features: 稀疏特征的配置信息，包含特征名和词汇表大小。
            dnn_hidden_units: 隐藏层形状
            regularization: 正则化系数
        """
        super(DNN, self).__init__()
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.dnn_hidden_units = dnn_hidden_units
        self.regularization = regularization

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

        # DNN隐藏层
        self.dnn_layers = []
        for units in self.dnn_hidden_units:
            self.dnn_layers.append(
                Dense(units, activation='relu', kernel_regularizer=l2(self.regularization))
            )

        # 输出层
        self.output_layer = Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=l2(self.regularization)
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
        dense_normalized_features = []
        for name in self.dense_features:
            dense_input = inputs[name]
            normalized = self.dense_normalization_layers[name](dense_input)
            dense_normalized_features.append(normalized)

        # 稀疏特征嵌入并展平
        sparse_features_output = []
        for feat in self.sparse_features:
            sparse_input = inputs[feat['name']]
            embedding = self.embedding_layers[feat['name']](sparse_input)
            sparse_features_output.append(Flatten()(embedding))

        # 合并稠密特征和稀疏特征
        combined_features = Concatenate()(dense_normalized_features + sparse_features_output)

        # 通过DNN隐藏层
        x = combined_features
        for layer in self.dnn_layers:
            x = layer(x)

        # 输出层
        output = self.output_layer(x)
        return output
