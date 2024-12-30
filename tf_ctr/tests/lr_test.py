# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import numpy as np
from tf_ctr.models.lr import LR

import get_data

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_lr_fit(self):
        X_train, X_test, y_train, y_test = get_data.get_criteo_data()

        dense_features, sparse_features = [], []
        dense_X, sparse_X = {}, {}
        for col in X_train.columns:
            x = X_train[col].values
            if col[0] == 'C':
                sparse_features.append({'name': col, 'vocab_size': len(set(x))})
                sparse_X[col] = x.reshape(-1, 1)
            elif col[0] == 'I':
                dense_features.append(col)
                dense_X[col] = x.reshape(-1, 1)
        X = {**dense_X, **sparse_X}
        y = y_train.values.reshape(-1, 1)

        for key, value in X.items():
            if not isinstance(value, (np.ndarray, tf.Tensor)):
                print(f"{key} is not an ndarray or Tensor, found type: {type(value)}")
            elif len(value.shape) != 2:
                print(f"{key} has invalid shape: {value.shape}")

        model = LR(dense_features=dense_features, sparse_features=sparse_features, embedding_dim=4)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
        model.fit(
            X, y,
            validation_split=0.3, shuffle=True, batch_size=32, epochs=10, verbose=2
        )
        model.summary()

        # 数据预处理：测试数据
        dense_X_test, sparse_X_test = {}, {}
        for col in X_test.columns:
            x = X_test[col].values
            if col[0] == 'C':
                sparse_X_test[col] = x.reshape(-1, 1)
            elif col[0] == 'I':
                dense_X_test[col] = x.reshape(-1, 1)

        X_test_processed = {**dense_X_test, **sparse_X_test}

        # 模型预测
        predictions = model.predict(X_test_processed)
        print("Predictions: ", predictions[:10])  # 输出前10个预测结果


if __name__ == '__main__':
    unittest.main()