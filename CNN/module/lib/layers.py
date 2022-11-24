import tensorflow as tf


class Real2Complex(tf.keras.layers.Layer):
    def __init__(self, phase=0.):
        super(Real2Complex, self).__init__()
        self.phase = tf.constant(phase, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "phase": self.phase.numpy()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return tf.complex(inputs, 0.) * tf.complex(tf.cos(self.phase), tf.sin(self.phase))

class ComplexDense(tf.keras.layers.Layer):

    def __init__(self, units=32, use_bias=False):
        super(ComplexDense, self).__init__()
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w_real = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True)
        self.w_imag = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True)

        if self.use_bias:
            self.b_real = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)
            self.b_imag = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        if self.use_bias:
            return tf.matmul(inputs, tf.complex(self.w_real, self.w_imag)) + tf.complex(self.b_real, self.b_imag)
        else:
            return tf.matmul(inputs, tf.complex(self.w_real, self.w_imag))


class MulConjugate(tf.keras.layers.Layer):
    def __init__(self):
        super(MulConjugate, self).__init__()

    def call(self, inputs):
        return inputs * tf.math.conj(inputs)


class Real(tf.keras.layers.Layer):
    def __init__(self):
        super(Real, self).__init__()

    def call(self, inputs):
        return tf.math.real(inputs)
