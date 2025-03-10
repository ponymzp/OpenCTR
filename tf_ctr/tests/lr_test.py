# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import numpy as np
from tf_ctr.models.lr import LR

from tf_ctr.tools import get_data, cal_model


class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_lr(self):
        for use_embedding in [True, False]:
            self.lr_sub(use_embedding)


    def lr_sub(self, use_embedding):
        X_train, X_test, y_train, y_test, dense_features, sparse_features = get_data.get_x_y_data(
            data_type='sample'
        )

        model = LR(
            dense_features=dense_features,
            sparse_features=sparse_features,
            use_embedding=use_embedding
        )

        cal_model.model_train(model, X_train, y_train)
        cal_model.model_predict(model,X_test, y_test)



if __name__ == '__main__':
    unittest.main()