import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

import losses


class AngularSpectrum(tf.keras.layers.Layer):
    def __init__(self, output_dim, wavelength=633e-9, z=0.0, d=1.0e-6, n=1.0, normalization=None, method=None):
        super(AngularSpectrum, self).__init__()
        self.output_dim = output_dim
        # self.wavelength = wavelength / n
        # self.k = 2 * np.pi / self.wavelength
        # self.z = z
        # self.d = d
        # self.n = n
        self.wavelength = wavelength
        self.wavelength_eff = wavelength / n
        self.k = 2 * np.pi / self.wavelength_eff
        self.z = z
        self.d = d
        self.n = n
        self.normalization = normalization if normalization is not None else "None"
        self.method = method if method is not None else "None"

        assert self.k >= 0.0
        assert self.z >= 0.0
        assert self.d > 0.0
        assert self.n > 0.0

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "wavelength": self.wavelength,
            "k": self.k,
            "z": self.z,
            "d": self.d,
            "n": self.n,
            "normalization": self.normalization,
            "method": self.method
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_dim):
        self.input_dim = input_dim

        width = self.input_dim[-1]
        height = self.input_dim[-2]
        u = np.fft.fftfreq(width, d=self.d)
        v = np.fft.fftfreq(height, d=self.d)
        UU, VV = np.meshgrid(u, v)
        w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength_eff ** 2, tf.sqrt(1 / self.wavelength_eff ** 2 - UU ** 2 - VV ** 2), 0).astype('float64')
        h = np.exp(1.0j * 2 * np.pi * w * self.z)

        if self.method == 'band_limited':
            du = 1 / (2 * width * self.d)
            dv = 1 / (2 * height * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_eff
            v_limit = 1 / (np.sqrt((2 * dv * self.z) ** 2 + 1)) / self.wavelength_eff
            u_filter = np.where(np.abs(UU) / (2 * u_limit) <= 1 / 2, 1, 0)
            v_filter = np.where(np.abs(VV) / (2 * v_limit) <= 1 / 2, 1, 0)
            h = h * u_filter * v_filter
        elif self.method == 'expand':
            self.pad_upper = math.ceil(self.input_dim[-2] / 2)
            self.pad_left = math.ceil(self.input_dim[-1] / 2)
            self.padded_width = int(input_dim[-1] + self.pad_left * 2)
            self.padded_height = int(input_dim[-2] + self.pad_upper * 2)

            u = np.fft.fftfreq(self.padded_width, d=self.d)
            v = np.fft.fftfreq(self.padded_height, d=self.d)

            du = 1 / (self.padded_width * self.d)
            dv = 1 / (self.padded_height * self.d)
            u_limit = 1 / (np.sqrt((2 * du * self.z) ** 2 + 1)) / self.wavelength_eff
            v_limit = 1 / (np.sqrt((2 * dv * self.z) ** 2 + 1)) / self.wavelength_eff
            UU, VV = np.meshgrid(u, v)

            u_filter = np.where(np.abs(UU) <= u_limit, 1, 0)
            v_filter = np.where(np.abs(VV) <= v_limit, 1, 0)

            w = np.where(UU ** 2 + VV ** 2 <= 1 / self.wavelength_eff ** 2, tf.sqrt(1 / self.wavelength_eff ** 2 - UU ** 2 - VV ** 2), 0).astype('float64')
            h = np.exp(1.0j * 2 * np.pi * w * self.z)
            h = h * u_filter * v_filter

        self.res = tf.cast(tf.complex(h.real, h.imag), dtype=tf.complex64)

    @tf.function
    def propagation(self, cximages):
        if self.method == 'band_limited':
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)
        elif self.method == 'expand':
            padding = [[0, 0], [self.pad_upper, self.pad_upper], [self.pad_left, self.pad_left]]
            images_pad = tf.pad(cximages, paddings=padding)
            images_pad_fft = tf.signal.fft2d(images_pad)
            u_images_pad = tf.signal.ifft2d(images_pad_fft * self.res)
            u_images = tf.keras.layers.Lambda(lambda x: x[:, self.pad_upper:self.pad_upper + self.input_dim[-2], self.pad_left:self.pad_left + self.input_dim[-1]])(u_images_pad)
            return u_images
        else:
            images_fft = tf.signal.fft2d(cximages)
            return tf.signal.ifft2d(images_fft * self.res)

    @tf.function
    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)

        u_rcp_x = self.propagation(rcp_x)
        u_lcp_x = self.propagation(lcp_x)

        rl = tf.stack([u_rcp_x, u_lcp_x], axis=1)

        if self.normalization == 'max':
            maximum = tf.reduce_max(tf.abs(rl))
            rl = rl / tf.complex(maximum, 0.0 * maximum)

        return rl


class CxTest(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(CxTest, self).__init__()
        self.output_dim = output_dim

    def call(self, inputs, **kwargs):
        return inputs ** 2


class ImageResizing(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(ImageResizing, self).__init__()
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim
        })
        return config

    def call(self, x):
        x_expnad = tf.image.resize(tf.expand_dims(x, -1), self.output_dim)
        x_expnad = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(x_expnad)
        return x_expnad


class ImageBinarization(tf.keras.layers.Layer):
    def __init__(self, threshold=0.5, minimum=0.0, maximum=1.0):
        super(ImageBinarization, self).__init__()
        self.threshold = threshold
        self.minimum = minimum
        self.maximum = maximum

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
            "minimum": self.minimum,
            "maximum": self.maximum
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        return tf.where(x >= self.threshold, self.maximum, self.minimum)


class IntensityToElectricField(tf.keras.layers.Layer):
    def __init__(self, output_dim, ini_theta=0.0):
        super(IntensityToElectricField, self).__init__()
        self.output_dim = output_dim
        self.ini_theta = ini_theta

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "ini_theta": self.ini_theta
        })
        return config

    def build(self, input_dim):
        self.input_dim = input_dim
        self.theta = tf.complex(self.ini_theta, 0.0)

    @tf.function
    def call(self, x):
        rcp_x = tf.complex(tf.sqrt(x / 2.0), 0.0 * x) * tf.exp(-1.0j * self.theta)
        lcp_x = tf.complex(tf.sqrt(x / 2.0), 0.0 * x) * tf.exp(1.0j * self.theta)
        return tf.stack([rcp_x, lcp_x], axis=1)


class ElectricFieldToIntensity(tf.keras.layers.Layer):
    def __init__(self, output_dim, normalization=None):
        super(ElectricFieldToIntensity, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "normalization": self.normalization
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, 0, :, :])(x)
        rcp_y = 1.0j * rcp_x
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, 0, :, :])(x)
        lcp_y = -1.0j * lcp_x

        tot_x = rcp_x + lcp_x
        tot_y = rcp_y + lcp_y

        intensity = tf.abs(tot_x) ** 2 / 2.0 + tf.abs(tot_y) ** 2 / 2.0

        if self.normalization == 'max':
            intensity = intensity / tf.reduce_max(intensity)

        return intensity


class MO(tf.keras.layers.Layer):
    def __init__(self, output_dim, limitation=None, theta=0.0, eta=0.0, kernel_regularizer=None):
        super(MO, self).__init__()
        self.output_dim = output_dim

        self.limitation = limitation if limitation is not None else "None"
        self.theta = theta
        self.eta = eta
        self.alpha_max = tf.complex(tf.constant(np.abs((np.log(1 + eta) - np.log(1 - eta))) / 2, dtype=tf.float32), 0.0)
        self.kernel_regularizer = kernel_regularizer
        assert len(self.output_dim) == 2
        assert -1.0 < self.eta < 1.0

    def build(self, input_dim):
        self.input_dim = input_dim
        self.mag = self.add_weight("magnetization",
                                   shape=[int(input_dim[-2]),
                                          int(input_dim[-1])],
                                   regularizer=self.kernel_regularizer)
        super(MO, self).build(input_dim)

    @tf.function
    def get_limited_theta(self):
        if self.limitation == 'tanh':
            return self.theta * tf.tanh(self.mag)
        elif self.limitation == 'sin':
            return self.theta * tf.sin(self.mag)
        elif self.limitation == 'sigmoid':
            return self.theta * (2.0 * tf.sigmoid(self.mag) - 1.0)
        else:
            return self.theta * self.mag

    @tf.function
    def get_limited_alpha(self):
        if self.limitation == 'tanh':
            eta_lim = self.eta * tf.tanh(self.mag)
            return -(tf.math.log(1.0 + eta_lim) - tf.math.log(1.0 - eta_lim)) / 2
        elif self.limitation == 'sin':
            eta_lim = self.eta * tf.sin(self.mag)
            return -(tf.math.log(1.0 + eta_lim) - tf.math.log(1.0 - eta_lim)) / 2
        elif self.limitation == 'sigmoid':
            eta_lim = self.eta * (2.0 * tf.sigmoid(self.mag) - 1.0)
            return -(tf.math.log(1.0 + eta_lim) - tf.math.log(1.0 - eta_lim)) / 2
        else:
            eta_lim = self.eta * self.mag
            return -(tf.math.log(1.0 + eta_lim) - tf.math.log(1.0 - eta_lim)) / 2

    @tf.function
    def get_limited_complex_faraday(self):
        theta = self.get_limited_theta()
        alpha = self.get_limited_alpha()
        return tf.complex(theta, -alpha)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "limitation": self.limitation,
            "theta": self.theta,
            "eta": self.eta
        })
        if self.kernel_regularizer:
            config.update({
                "reguralizer": self.kernel_regularizer.get_config()
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        phi = self.get_limited_complex_faraday()

        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)

        rcp_x_mo = rcp_x * tf.exp(-self.alpha_max) * tf.exp(-1.0j * phi)
        lcp_x_mo = lcp_x * tf.exp(-self.alpha_max) * tf.exp(1.0j * phi)

        return tf.stack([rcp_x_mo, lcp_x_mo], axis=1)


class MNISTDetector(tf.keras.layers.Layer):
    def __init__(self, output_dim, inverse=False, activation=None, normalization=None, **kwargs):
        super(MNISTDetector, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.inverse = inverse
        self.activation = activation
        self.normalization = normalization

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "inverse": self.inverse,
            "activation": self.activation,
            "normalization": self.normalization
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.input_dim = input_shape
        width = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))
        height = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))

        w0 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w0[2 * height:3 * height, width:2 * width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w1[2 * height:3 * height, 4 * width:5 * width] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w2[2 * height:3 * height, 7 * width:8 * width] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w3[4 * height:5 * height, 1 * width:2 * width] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w4[4 * height:5 * height, 3 * width:4 * width] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w5[4 * height:5 * height, 5 * width:6 * width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w6[4 * height:5 * height, 7 * width:8 * width] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w7[6 * height:7 * height, width:2 * width] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w8[6 * height:7 * height, 4 * width:5 * width] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        w9[6 * height:7 * height, 7 * width:8 * width] = 1.0
        w9 = tf.constant(w9)

        if self.inverse:
            self.filter = -tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)
        else:
            self.filter = tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)

    def call(self, x, **kwargs):
        y = tf.tensordot(x, self.filter, axes=[[1, 2], [0, 1]])

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        return y


class CircleOnCircumferenceDetector(tf.keras.layers.Layer):
    def __init__(self, output_dim, r1, r2, activation=None, normalization=None, name="circle_on_circumference_detector", **kwargs):
        super(CircleOnCircumferenceDetector, self).__init__(name=name, **kwargs)
        assert 0 < r1
        assert 0 < output_dim
        assert 0 < r2 < r1 * np.tan(2 * np.pi / (2 * output_dim))
        assert 0 < r1 + r2 < np.max(output_dim) / 2
        self.output_dim = output_dim
        self.r1 = r1
        self.r2 = r2
        self.activation = activation
        self.normalization = normalization

    @staticmethod
    def make_filters(shape, r1, r2, class_num):
        rads = np.linspace(0, 2 * np.pi, class_num, endpoint=False)
        x = np.arange(shape[1])
        y = np.arange(shape[0])
        xx, yy = np.meshgrid(x, y)
        xx = xx - np.mean(xx)
        yy = yy - np.mean(yy)
        f_list = []
        for rad in rads:
            p = r1 * np.cos(rad - np.pi / 2)
            q = r1 * np.sin(rad - np.pi / 2)
            f_list.append(np.where((xx - p) ** 2 + (yy - q) ** 2 <= r2 ** 2, 1, 0))
        return tf.constant(np.array(f_list), dtype=tf.float32)

    @staticmethod
    def plot(shape, r1, r2, class_num, ax=None):
        losses.CategoricalCircleOnCircumferenceMSE.plot(shape, r1, r2, class_num, ax)

    @tf.function
    def get_photo_mask(self):
        return tf.reduce_sum(self.filters, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "r1": self.r1,
            "r2": self.r2,
            "activation": self.activation,
            "normalization": self.normalization
        })
        return config

    def build(self, input_dim):
        self.input_dim = input_dim

        self.filters = losses.CategoricalCircleOnCircumferenceMSE.make_filters(self.input_dim, self.r1, self.r2, self.output_dim)

    def call(self, x):
        y = tf.tensordot(x, self.filter, axes=[[1, 2], [1, 2]])

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        return y


class MNISTFilter(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(MNISTFilter, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "activation": self.activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.input_dim = input_shape
        width = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))
        height = min(int(tf.floor(self.input_dim[2] / 9.0)), int(tf.floor(self.input_dim[1] / 7.0)))

        filter = np.zeros((self.input_dim[-2], self.input_dim[-1]), dtype='float32')
        filter[2 * height:3 * height, width:2 * width] = 1.0
        filter[2 * height:3 * height, 4 * width:5 * width] = 1.0
        filter[2 * height:3 * height, 7 * width:8 * width] = 1.0
        filter[4 * height:5 * height, 1 * width:2 * width] = 1.0
        filter[4 * height:5 * height, 3 * width:4 * width] = 1.0
        filter[4 * height:5 * height, 5 * width:6 * width] = 1.0
        filter[4 * height:5 * height, 7 * width:8 * width] = 1.0
        filter[6 * height:7 * height, width:2 * width] = 1.0
        filter[6 * height:7 * height, 4 * width:5 * width] = 1.0
        filter[6 * height:7 * height, 7 * width:8 * width] = 1.0

        self.filter = tf.constant(filter)

    def call(self, x, **kwargs):
        return tf.multiply(x, self.filter)


class FaradayRotationByStokes(tf.keras.layers.Layer):
    def __init__(self, output_dim, normalization=None, eps=1.0e-20):
        super(FaradayRotationByStokes, self).__init__()
        self.output_dim = output_dim
        self.normalization = normalization
        self.eps = eps

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "normalization": self.normalization,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, **kwargs):
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        rcp_y = 1.0j * rcp_x
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)
        lcp_y = -1.0j * lcp_x

        E0 = rcp_x + lcp_x
        I0 = tf.abs(E0) ** 2 / 2.0
        E90 = rcp_y + lcp_y
        I90 = tf.abs(E90) ** 2 / 2.0
        E45_x = (rcp_x + rcp_y + lcp_x + lcp_y) / 2.0
        E45_y = (rcp_x + rcp_y + lcp_x + lcp_y) / 2.0
        I45 = tf.abs(E45_x) ** 2 / 2 + tf.abs(E45_y) ** 2 / 2.0
        E135_x = (rcp_x - rcp_y + lcp_x - lcp_y) / 2.0
        E135_y = (-rcp_x + rcp_y - lcp_x + lcp_y) / 2.0
        I135 = tf.abs(E135_x) ** 2 / 2 + tf.abs(E135_y) ** 2 / 2.0

        S1 = I0 - I90
        S2 = I45 - I135

        # theta = tf.where(S1**2 > self.eps, tf.atan(S2*S1 / S1**2) / 2.0, tf.atan(S2*S1 / self.eps) / 2.0,)
        theta = tf.atan(S2 * S1 / (S1 ** 2 + self.eps)) / 2.0

        if self.normalization == 'minmax':
            minimum = tf.reduce_min(theta, axis=[1, 2], keepdims=True)
            maximum = tf.reduce_max(theta, axis=[1, 2], keepdims=True)
            theta = (theta - minimum) / (maximum - minimum)

        return theta


class Polarizer(tf.keras.layers.Layer):
    def __init__(self, output_dim, phi=0.0, trainable=False):
        super(Polarizer, self).__init__()
        self.output_dim = output_dim
        self.phi = tf.Variable(phi, name="phi", trainable=trainable)
        self.trainable = trainable

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "phi": self.phi.numpy(),
            "trainable": self.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, 0, :, :])(x)
        rcp_y = 1.0j * rcp_x
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, 0, :, :])(x)
        lcp_y = -1.0j * lcp_x

        p00 = tf.complex(tf.cos(self.phi) ** 2.0, 0.0)
        p01 = tf.complex(tf.sin(2.0 * self.phi) / 2.0, 0.0)
        p10 = p01
        p11 = tf.complex(tf.sin(self.phi) ** 2.0, 0.0)

        rcp_x_pol = p00 * rcp_x + p01 * rcp_y
        rcp_y_pol = p10 * rcp_x + p11 * rcp_y

        lcp_x_pol = p00 * lcp_x + p01 * lcp_y
        lcp_y_pol = p10 * lcp_x + p11 * lcp_y

        rcp = tf.stack([rcp_x_pol, rcp_y_pol], axis=1)
        lcp = tf.stack([lcp_x_pol, lcp_y_pol], axis=1)

        rl = tf.stack([rcp, lcp], axis=1)

        return rl


class Dielectric(tf.keras.layers.Layer):
    def __init__(self, tensor):
        super(Dielectric, self).__init__()
        self.tensor = tf.complex(tensor, 0.0 * tensor)

    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, 0, :, :])(x)
        rcp_y = tf.keras.layers.Lambda(lambda x: x[:, 0, 1, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, 0, :, :])(x)
        lcp_y = tf.keras.layers.Lambda(lambda x: x[:, 1, 1, :, :])(x)


class GGG(AngularSpectrum):
    def __init__(self, output_dim, wavelength, z=0.0, d=1.0e-6, normalization=None, method=None):
        super(GGG, self).__init__(output_dim, wavelength, z=z, d=d, n=2.0, normalization=normalization, method=method)


class FaradayRotationByArgument(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(FaradayRotationByArgument, self).__init__()
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def calc_argument(self, cmpx):
        real = tf.math.real(cmpx)
        imag = tf.math.imag(cmpx)

        arg = tf.where(tf.not_equal(imag, 0.0), 2.0 * tf.atan((tf.sqrt(real ** 2 + imag ** 2) - real) / imag), 0.0)
        arg = tf.where((real > 0.0) & (tf.equal(imag, 0.0)), 0.0, arg)
        arg = tf.where((real < 0.0) & tf.equal(imag, 0.0), np.pi, arg)
        arg = tf.where(tf.equal(real, 0.0) & tf.equal(imag, 0.0), 0.0, arg)

        return arg

    def call(self, x):
        rcp_x = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(x)
        lcp_x = tf.keras.layers.Lambda(lambda x: x[:, 1, :, :])(x)

        rcp_arg = self.calc_argument(rcp_x)
        lcp_arg = self.calc_argument(lcp_x)

        delta_phi = (rcp_arg - lcp_arg) / 2.0

        return delta_phi


class PhaseToPeriodic(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(PhaseToPeriodic, self).__init__()
        self.output_dim = output_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        return tf.sin(x)


class Softmax(tf.keras.layers.Layer):
    def __init__(self, eps=0.0):
        super(Softmax, self).__init__()
        self.eps = tf.Variable(eps, trainable=False, name="epsilon")

    def get_config(self):
        config = super().get_config()
        config.update({
            "eps": self.eps.numpy()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        minimum = tf.reduce_min(x, axis=-1, keepdims=True)
        return tf.nn.softmax(x - minimum, axis=-1)


class MinMaxNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(MinMaxNormalization, self).__init__()

    def call(self, x):
        maximum = tf.reduce_max(x)
        minimum = tf.reduce_min(x)
        return tf.nn.softmax(x, axis=-1) + self.eps


class MNISTDifferentialDetector(tf.keras.layers.Layer):
    def __init__(self, output_dim, inverse=False, activation=None, normalization=None, **kwargs):
        super(MNISTDifferentialDetector, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.inverse = inverse
        self.activation = activation
        self.normalization = normalization

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "inverse": self.inverse,
            "activation": self.activation,
            "normalization": self.normalization
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def make_positive_filter(input_shape):
        width = int(input_shape[-1] / 10)
        height = min(int(input_shape[-2] / 2 / 5), width)
        pad_width = int(np.round(width / 2))
        pad_height = int(np.round((input_shape[-1] / 2 - height * 3) / 2))

        w0 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w0[pad_height:pad_height + height, pad_width:pad_width + width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w1[pad_height:pad_height + height, pad_width + width * 2:pad_width + width * 3] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w2[pad_height:pad_height + height, pad_width + width * 4:pad_width + width * 5] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w3[pad_height:pad_height + height, pad_width + width * 6:pad_width + width * 7] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w4[pad_height:pad_height + height, pad_width + width * 8:pad_width + width * 9] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w5[pad_height + height * 2:pad_height + height * 3, pad_width:pad_width + width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w6[pad_height + height * 2:pad_height + height * 3, pad_width + width * 2:pad_width + width * 3] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w7[pad_height + height * 2:pad_height + height * 3, pad_width + width * 4:pad_width + width * 5] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w8[pad_height + height * 2:pad_height + height * 3, pad_width + width * 6:pad_width + width * 7] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w9[pad_height + height * 2:pad_height + height * 3, pad_width + width * 8:pad_width + width * 9] = 1.0
        w9 = tf.constant(w9)

        return tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)

    @staticmethod
    def make_negative_filter(input_shape):
        width = int(input_shape[-1] / 10)
        height = min(int(input_shape[-2] / 2 / 5), width)

        margin_height = int(input_shape[-2] / 2)

        pad_width = int(np.round(width / 2))
        pad_height = int(np.round((input_shape[-1] / 2 - height * 3) / 2)) + margin_height

        w0 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w0[pad_height:pad_height + height, pad_width:pad_width + width] = 1.0
        w0 = tf.constant(w0)

        w1 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w1[pad_height:pad_height + height, pad_width + width * 2:pad_width + width * 3] = 1.0
        w1 = tf.constant(w1)

        w2 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w2[pad_height:pad_height + height, pad_width + width * 4:pad_width + width * 5] = 1.0
        w2 = tf.constant(w2)

        w3 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w3[pad_height:pad_height + height, pad_width + width * 6:pad_width + width * 7] = 1.0
        w3 = tf.constant(w3)

        w4 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w4[pad_height:pad_height + height, pad_width + width * 8:pad_width + width * 9] = 1.0
        w4 = tf.constant(w4)

        w5 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w5[pad_height + height * 2:pad_height + height * 3, pad_width:pad_width + width] = 1.0
        w5 = tf.constant(w5)

        w6 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w6[pad_height + height * 2:pad_height + height * 3, pad_width + width * 2:pad_width + width * 3] = 1.0
        w6 = tf.constant(w6)

        w7 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w7[pad_height + height * 2:pad_height + height * 3, pad_width + width * 4:pad_width + width * 5] = 1.0
        w7 = tf.constant(w7)

        w8 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w8[pad_height + height * 2:pad_height + height * 3, pad_width + width * 6:pad_width + width * 7] = 1.0
        w8 = tf.constant(w8)

        w9 = np.zeros((input_shape[-2], input_shape[-1]), dtype='float32')
        w9[pad_height + height * 2:pad_height + height * 3, pad_width + width * 8:pad_width + width * 9] = 1.0
        w9 = tf.constant(w9)

        return tf.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8, w9], axis=-1)

    @staticmethod
    def plot(input_shape):
        positive = MNISTDifferentialDetector.make_positive_filter(input_shape)
        negative = MNISTDifferentialDetector.make_negative_filter(input_shape)
        image = tf.reduce_sum(positive, axis=-1) + -1 * tf.reduce_sum(negative, axis=-1)
        plt.imshow(image)

    def build(self, input_shape):
        self.input_dim = input_shape
        self.positive_filter = self.make_positive_filter(input_shape)
        self.negative_filter = self.make_negative_filter(input_shape)

    def call(self, x, **kwargs):
        y_positive = tf.tensordot(x, self.positive_filter, axes=[[1, 2], [0, 1]])
        y_negative = tf.tensordot(x, self.negative_filter, axes=[[1, 2], [0, 1]])

        y = y_positive - y_negative

        if self.normalization == 'minmax':
            maximum = tf.reduce_max(y)
            minimum = tf.reduce_min(y)
            y = (y - minimum) / (maximum - minimum)

        if self.activation == 'softmax':
            y = tf.nn.softmax(y)

        return y