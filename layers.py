import tensorflow as tf
from tensorflow_core.python.keras.layers import Lambda
import numpy as np
import matplotlib.pyplot as plt


class CxMO(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(CxMO, self).__init__()
        self.output_dim = output_dim

    # input[0,:,:] = real
    # input[1,:,:] = image
    def build(self, input_dim):
        self.phi = self.add_variable("phi",
                                     shape=[int(input_dim[-2]),
                                            int(input_dim[-1])])

    def call(self, x):
        x_real = Lambda(lambda x: x[0, :, :], output_shape=self.output_dim)(x)  # real
        x_imag = Lambda(lambda x: x[1, :, :], output_shape=self.output_dim)(x)  # imag
        mo_real = tf.cos(self.phi)
        mo_imag = tf.sin(self.phi)

        real = tf.matmul(x_real, mo_real) - tf.matmul(x_imag, mo_imag)
        imag = tf.matmul(x_real, mo_imag) + tf.matmul(x_imag, mo_real)
        tf.stack([real, imag], axis=0)
        cmpx = tf.stack([real, imag], axis=0)
        return cmpx


class FreeSpacePropagation(tf.keras.layers.Layer):
    def __init__(self, output_dim, pitch_size, z, k):
        super(FreeSpacePropagation, self).__init__()
        self.output_dim = output_dim
        self.pitch_size = pitch_size
        self.z = z
        self.k = k

    def build(self, input_shape):
        x1 = np.arange(0, input_shape[2], 1)
        y1 = np.arange(0, input_shape[1], 1)
        xx1, yy1 = np.meshgrid(x1, y1)
        xx1 = xx1.reshape(1, -1)
        yy1 = yy1.reshape(1, -1)

        x2 = np.arange(0, self.output_dim[1], 1)
        y2 = np.arange(0, self.output_dim[0], 1)
        xx2, yy2 = np.meshgrid(x2, y2)
        xx2 = xx2.reshape(-1, 1)
        yy2 = yy2.reshape(-1, 1)

        dx = self.pitch_size*(xx1 - xx2)
        dy = self.pitch_size*(yy1 - yy2)
        r = np.sqrt(dx**2 + dy**2 + self.z**2)
        w = 1/(2*np.pi) * self.z / r * (1/r - 1j*self.k) * np.exp(1j * self.k * r)

        self.w_real = tf.Variable(initial_value=w.real.astype('float32'),
                                              trainable=False)
        self.w_imag = tf.Variable(initial_value=w.imag.astype('float32'),
                                              trainable=False)

    def call(self, x, **kwargs):
        x_real = Lambda(lambda x: x[0, :, :], output_shape=self.output_dim)(x)  # real
        x_imag = Lambda(lambda x: x[1, :, :], output_shape=self.output_dim)(x)  # imag
        x_real = tf.reshape(x_real, (-1, 1))
        x_imag = tf.reshape(x_imag, (-1, 1))
        real = tf.matmul(self.w_real, x_real) - tf.matmul(self.w_imag, x_imag)
        imag = tf.matmul(self.w_real, x_imag) + tf.matmul(self.w_imag, x_real)
        real = tf.reshape(real, self.output_dim)
        imag = tf.reshape(imag, self.output_dim)
        cmpx = tf.concat([[real], [imag]], axis=0)
        return cmpx


if __name__ == '__main__':
    l = 633e-9
    k = np.pi * 2 / l
    shape = (28, 28)
    pattern = np.zeros(shape)
    pattern[0::3, 0::3] = 1
    cxpatter = np.exp(1j*pattern)
    patten_real = cxpatter.real
    patten_imag = cxpatter.imag
    layer = FreeSpacePropagation(shape, 1e-6, 1e-5, k)
    out = layer(np.stack([patten_real, patten_imag], axis=0))
    print(out.shape)

    plt.imshow(out[0, :, :])
    plt.show()
