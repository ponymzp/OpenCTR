# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import numpy as np
from tf_ctr.models.dnn import DNN

from tf_ctr.tools import get_data, cal_model


class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_dnn(self):
        for dnn_hidden_units in [(16, 2), (128,64)]:
            self.dnn_sub(dnn_hidden_units)


    def dnn_sub(self, dnn_hidden_units):
        X_train, X_test, y_train, y_test, dense_features, sparse_features = get_data.get_x_y_data(
            data_type='sample'
        )

        model = DNN(
            dense_features=dense_features,
            sparse_features=sparse_features,
            dnn_hidden_units=dnn_hidden_units
        )

        cal_model.model_train(model, X_train, y_train)
        cal_model.model_predict(model,X_test, y_test)



if __name__ == '__main__':
    unittest.main()