import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Embedding, Concatenate

from sklearn.preprocessing import LabelEncoder


class LR(tf.keras.Model):
    def __init__(self, dense_features, sparse_features, embedding_dim=8):
        """
        初始化模型
        Args:
            dense_features: 稠密特征的名称列表。
            sparse_features: 稀疏特征的配置信息，包含特征名和词汇表大小。
            embedding_dim: Embedding 层的维度。
        """
        super(LR, self).__init__()
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.embedding_dim = embedding_dim

        # Embedding 层
        self.embedding_layers = {
            feat['name']: Embedding(
                input_dim=feat['vocab_size'],
                output_dim=self.embedding_dim,
                input_length=1,
                name=f"{feat['name']}_embedding"
            ) for feat in self.sparse_features
        }

        # 稠密特征归一化层
        self.dense_normalization_layers = {
            name: tf.keras.layers.LayerNormalization(axis=-1) for name in self.dense_features
        }

        # 输出层
        self.output_layer = Dense(1, activation="sigmoid")


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
        sparse_embeddings = []
        for feat in self.sparse_features:
            sparse_input = inputs[feat['name']]
            embedding = self.embedding_layers[feat['name']](sparse_input)
            sparse_embeddings.append(Flatten()(embedding))

        # 合并稠密特征和稀疏特征
        combined_features = Concatenate()(dense_normalized_features + sparse_embeddings)

        # 输出层
        output = self.output_layer(combined_features)
        return output