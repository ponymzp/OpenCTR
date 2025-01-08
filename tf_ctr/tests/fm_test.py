# -*- coding: utf-8 -*-

import unittest
import tensorflow as tf
import numpy as np
from tf_ctr.models.fm import FM

from tf_ctr.tools import get_data, cal_model


class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_fm(self):
        self.fm_sub()


    def fm_sub(self):
        X_train, X_test, y_train, y_test, dense_features, sparse_features = get_data.get_x_y_data(
            data_type='sample'
        )

        model = FM(
            dense_features=dense_features,
            sparse_features=sparse_features
        )

        cal_model.model_train(model, X_train, y_train)
        cal_model.model_predict(model,X_test, y_test)



if __name__ == '__main__':
    unittest.main()