import tensorflow as tf
import numpy as np


class CxTest(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(CxTest, self).__init__()
        self.output_dim = output_dim

    def call(self, inputs, **kwargs):

        return inputs