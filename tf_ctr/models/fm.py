# -*- coding:utf-8 -*-
"""
FM模型
Author:
    Pony Ma, ponymzp@163.com
References
    - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Embedding, Concatenate, LayerNormalization
from tensorflow.keras.regularizers import l2

class FM(tf.keras.Model):
    def __init__(self, dense_features, sparse_features, regularization=0.001):
        """
        初始化模型
        Args:
            dense_features: 稠密特征的名称列表。
            sparse_features: 稀疏特征的配置信息，包含特征名和词汇表大小。
            regularization: 正则化系数
        """
        super(FM, self).__init__()
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.regularization = regularization

        # 稠密特征归一化层
        self.dense_normalization_layers = {
            name: LayerNormalization(axis=-1) for name in self.dense_features
        }

        # Embedding 层
        self.embedding_layers = {
            feat['name']: Embedding(
                input_dim=feat['vocab_size'],
                output_dim=feat['embedding_dim'],
                input_length=1,
                embeddings_regularizer=l2(self.regularization),
                name=f"{feat['name']}_embedding"
            ) for feat in self.sparse_features
        }

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
        dense_normalized_features = [
            self.dense_normalization_layers[name](inputs[name])
            for name in self.dense_features
        ]

        # 稀疏特征嵌入并展平
        sparse_embeddings = [
            self.embedding_layers[feat['name']](inputs[feat['name']])
            for feat in self.sparse_features
        ]
        sparse_embeddings_flattened = [Flatten()(embedding) for embedding in sparse_embeddings]

        # 一阶特征（线性部分）
        linear_terms = Concatenate()(dense_normalized_features + sparse_embeddings_flattened)

        # 二阶特征交互部分
        summed_embeddings = tf.reduce_sum(sparse_embeddings, axis=0)
        summed_embeddings_square = tf.square(summed_embeddings)

        squared_embeddings = [tf.square(embedding) for embedding in sparse_embeddings]
        squared_sum_embeddings = tf.reduce_sum(squared_embeddings, axis=0)

        cross_terms = 0.5 * tf.reduce_sum(summed_embeddings_square - squared_sum_embeddings, axis=-1, keepdims=True)
        cross_terms = Flatten()(cross_terms)

        # 合并一阶和二阶特征
        combined_features = Concatenate()([linear_terms, cross_terms])

        # 输出层
        output = self.output_layer(combined_features)
        return output