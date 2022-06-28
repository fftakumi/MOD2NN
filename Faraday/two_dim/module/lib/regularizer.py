import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Custom', name='shift_l1')
class ShiftL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=0., shift=0.):
        self.l1 = l1
        self.shift = shift

    def __call__(self, x):
        return self.l1 * tf.math.reduce_sum(tf.math.abs(self.shift - tf.math.abs(x)))

    def get_config(self):
        return {'l1': float(self.l1), 'shift': float(self.shift)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='shift_l2')
class ShiftL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l2=0., shift=0.):
        self.l1 = l2
        self.shift = shift

    def __call__(self, x):
        return self.l1 * tf.math.reduce_sum((self.shift**2 - x**2))

    def get_config(self):
        return {'l2': float(self.l1), 'shift': float(self.shift)}
